"""Population R6 error explanation on the same verified predictions as the streak follow-up."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()
FOLLOWUP = ROOT / "outputs/t1-r6-followup-20261110"
REFERENCE = ROOT / "outputs/timing-alignment-population-gpu-20261028T133000Z"
sys.path.insert(0, str(ROOT / "src"))
from eeg_kinematics.error_dynamics import groups, scores, STAGES, SPECS, WINDOWS, derivatives, trial_weights

spec = importlib.util.spec_from_file_location("r6_event_alignment", FOLLOWUP / "r6_population_followup.py")
aligned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aligned)
PEOPLE = [f"P{i}" for i in range(1, 13)]
LABELS = {"ridge": "Ridge", "cnn_elu": "CNN + ELU"}
COLORS = {"ridge": "#3b6b83", "cnn_elu": "#ac5c27"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def features_and_errors(person, entry, expected):
    participant = aligned.raw_struct(aligned.RAW_DATA_DIR / person / f"{person}_AllLifts.mat", "P")
    columns = {str(name): i for i, name in enumerate(participant.ColNames)}
    kin = loadmat(aligned.REDUCED_DATA_DIR / f"kin_{person}.mat")["kin"].astype(float)
    bounds = entry["series_bounds"]
    center, scale = (np.array(entry["scaling"][key]) for key in ("target_mean", "target_scale"))
    with np.load(REFERENCE / person / "split_indices.npz") as f:
        t, ids = f["test_targets"], f["test_series_ids"]
    stage, trial = np.zeros(len(t), dtype=int), np.zeros(len(t), dtype=int)
    motion = {w: np.zeros((len(t), 3, 3)) for w in WINDOWS}
    for series in (8, 9):
        lo, hi = bounds[str(series)]
        stages, trials, coverage = aligned.stage_alignment(person, series, hi-lo, participant.AllLifts, columns)
        keep = ids == series
        local = t[keep] - lo
        inside = (local >= 40) & (local < hi-lo-40) & (trials[local] > 0)
        stage[keep] = stages[local]
        trial[keep] = np.where(inside, series*1000 + trials[local], 0)
        assert {**coverage, "targets": int(keep.sum()), "trial_targets": int(inside.sum()),
                "contributing_trials": int(np.unique(trials[local][inside]).size)} == expected[f"{person}/{series}"]
        for w in WINDOWS:
            motion[w][keep] = derivatives(kin[lo:hi], w)[local]
    pred = {}
    for model, fits in aligned.MODELS.items():
        pred[model] = []
        for fit in fits:
            with np.load(REFERENCE / person / f"{fit}_predictions.npz") as z:
                np.testing.assert_array_equal(t, z["target_sample_index"])
                np.testing.assert_array_equal(ids, z["series_id"])
                np.testing.assert_allclose(kin[t], z["y_true"]*scale+center, atol=1e-7, rtol=1e-6)
                pred[model].append(z["y_pred"]*scale+center)
    use = trial > 0
    anchors = [np.argsort(-np.histogram(kin[:bounds["7"][0], axis], aligned.EDGES)[0], kind="stable")[:5]
               for axis in range(3)]
    return kin[t][use], ids[use], stage[use], trial[use], {w:x[use] for w,x in motion.items()}, {
        m:[x[use] for x in pp] for m,pp in pred.items()
    }, anchors


def fit_models(y, x, trial, series):
    prediction = np.empty(len(y))
    for origin, destination in ((8, 9), (9, 8)):
        train, test = series == origin, series == destination
        w = trial_weights(trial[train])
        if x is None:
            prediction[test] = np.average(y[train], weights=w)
            continue
        scaler = StandardScaler().fit(x[train], sample_weight=w)
        model = Ridge(alpha=10, solver="svd").fit(scaler.transform(x[train]), y[train], sample_weight=w)
        design = scaler.transform(x[test])
        prediction[test] = model.predict(design)
        np.testing.assert_allclose(prediction[test], design @ model.coef_ + model.intercept_, atol=1e-12)
    return scores(y, prediction, trial)


def average(values):
    v = [x for x in values if x is not None]
    return float(np.mean(v)) if v else None


def collect(reference, alignment):
    fits, speed_stats, correlations = [], [], []
    for person in PEOPLE:
        position, series, stage, trial, motion, predictions, anchors = features_and_errors(
            person, reference["participant_results"][person], alignment
        )
        for window, deriv in motion.items():
            pieces = groups(position, stage, deriv)
            designs = {key:np.column_stack([pieces[g] for g in groupspec]) if groupspec else None
                       for key, groupspec in SPECS.items()}
            speed = np.linalg.norm(deriv[:, 0], axis=1)
            for model, seed_predictions in predictions.items():
                ys = [np.linalg.norm(pred-position, axis=1) for pred in seed_predictions]
                for name,x in designs.items():
                    metrics = [fit_models(y, x, trial, series) for y in ys]
                    fits.append(dict(participant=person, model=model, window_ms=window*2, features=name,
                                     n=len(trial), trials=len(np.unique(trial)),
                                     mse=average([m["mse"] for m in metrics]),
                                     r2=average([m["r2_error_target"] for m in metrics])))
                if window != 41:
                    continue
                for axis,label in enumerate("XYZ"):
                    masks = [aligned.regions(position[:, axis], p[:, axis], anchors[axis])[0]
                             for p in seed_predictions]
                    for code, name in enumerate(("near diagonal", "vertical", "horizontal")):
                        values = [float(np.median(speed[m==code])) for m in masks
                                  if np.unique(trial[m==code]).size >= 5]
                        if len(values)==len(seed_predictions):
                            speed_stats.append(dict(participant=person,model=model,axis=label,region=name,
                                                    median_speed=average(values)))
                for i,name in enumerate(STAGES[1:], start=1):
                    select = stage==i
                    count = np.unique(trial[select]).size
                    if count<5 or np.ptp(speed[select])==0:
                        continue
                    values = [float(spearmanr(speed[select],y[select]).statistic) for y in ys
                              if np.ptp(y[select])>0]
                    if len(values)==len(ys):
                        correlations.append(dict(participant=person,model=model,stage=name,n=int(select.sum()),
                                                 trials=count,rho=average(values)))
        print("CHECKED",person,flush=True)
    assert len(fits) == len(PEOPLE)*len(LABELS)*len(WINDOWS)*len(SPECS)
    return fits, speed_stats, correlations


def summarize(fits, speed, corr):
    draws = np.random.default_rng(20261111).integers(0,12,(10000,12))
    contrast, speeds, stages, features = {}, {}, {}, {}
    for model in LABELS:
        contrast[model], features[model] = {}, {}
        for window in (42,82,162):
            rows = [r for r in fits if r["model"]==model and r["window_ms"]==window]
            def vector(name, metric):
                chosen = [r for r in rows if r["features"]==name]
                assert [r["participant"] for r in chosen] == PEOPLE
                return np.array([r[metric] for r in chosen])
            baseline, full = vector("position_stage","mse"), vector("combined","mse")
            delta, rel = full-baseline, 100*(1-full/baseline)
            def ci(values):
                return np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist()
            contrast[model][str(window)] = dict(baseline=float(baseline.mean()),full=float(full.mean()),
                delta=float(delta.mean()),delta_ci=ci(delta),improved=int((delta<0).sum()),
                relative=float(rel.mean()),relative_ci=ci(rel))
            if window == 82:
                features[model] = {k:dict(mse=float(vector(k,"mse").mean()),r2=float(vector(k,"r2").mean()))
                                   for k in SPECS}
        speeds[model] = {}
        for region in ("near diagonal","vertical","horizontal"):
            speeds[model][region] = {}
            for axis in "XYZ":
                vals = [r["median_speed"] for r in speed if (r["model"],r["region"],r["axis"])==(model,region,axis)]
                speeds[model][region][axis] = dict(n=len(vals),median=float(np.median(vals)) if vals else None)
        stages[model] = {}
        for stage in STAGES[1:]:
            vals = [r["rho"] for r in corr if (r["model"],r["stage"])==(model,stage)]
            stages[model][stage] = dict(n=len(vals),mean=average(vals))
    return dict(contrast=contrast,speeds=speeds,stages=stages,features=features)


def plots(out, summary, correlations):
    plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False})
    fig, ax = plt.subplots(figsize=(10.7,4.0),layout="constrained")
    ax.axhline(0,color="#677581",ls="--",lw=1)
    for j,model in enumerate(LABELS):
        x = np.arange(3)+(j-.5)*.16
        v = [summary["contrast"][model][str(w)] for w in (42,82,162)]
        mean = np.array([r["relative"] for r in v])
        lo,hi = np.array([r["relative_ci"] for r in v]).T
        ax.errorbar(x,mean,yerr=[mean-lo,hi-mean],fmt="o",color=COLORS[model],label=LABELS[model],capsize=4,lw=2)
    ax.set(xticks=range(3),xticklabels=["42 ms","82 ms (primary)","162 ms"],
           xlabel="Smoothing window",ylabel="Reduction in error-model MSE (%)")
    ax.legend(frameon=False,ncol=2);ax.grid(axis="y",alpha=.2)
    fig.savefig(out/"smoothing_comparison.png",dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11.1,4.3),sharex=True,sharey=True,layout="constrained")
    labels=["Preparation","Reach","Preload","Loading","Airborne before cue","After cue","Release","Return"]
    for ax,model in zip(axs,LABELS):
        ax.axvline(0,color="#677581",ls="--",lw=1)
        for i,stage in enumerate(STAGES[1:]):
            data = [r["rho"] for r in correlations if r["model"]==model and r["stage"]==stage]
            if data:
                ax.scatter(data,np.full(len(data),i),color=COLORS[model],alpha=.31,s=23)
                ax.scatter(np.mean(data),i,color=COLORS[model],marker="D",s=45,zorder=3)
            ax.text(.96,i,f"n={len(data)}",ha="right",va="center",fontsize=9,color="#526778")
        ax.set(title=LABELS[model],xlabel="Speed–error Spearman correlation",xlim=(-1,1),yticks=range(8),yticklabels=labels)
        ax.grid(axis="x",alpha=.2)
    axs[0].invert_yaxis()
    fig.savefig(out/"stage_speed_associations.png",dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    out=parser.parse_args().output.resolve()
    out.mkdir(exist_ok=False)
    for name in ("verification.json","population_verification.json"):
        assert read(REFERENCE/name)["status"]=="passed"
    alignment = read(FOLLOWUP/"alignment.json")
    reference = read(REFERENCE/"results.json")
    fits,speed,corr = collect(reference,alignment)
    summary = summarize(fits,speed,corr)
    write_csv(out/"participant_error_models.csv",fits)
    write_csv(out/"participant_streak_speeds.csv",speed)
    write_csv(out/"participant_stage_correlations.csv",corr)
    (out/"population_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    plots(out,summary,corr)
    report = [
        "# R6 — Population error explanation", "",
        "P1–P12, ridge and CNN + ELU saved 2 ms wrist predictions, 36,811 event-aligned targets in 804 trials. "
        "The event coverage reproduces the existing population streak analysis. Kinematic errors use reduced coordinates.", "",
        "For each participant and decoder, error explainers fit on series 8 and score on 9, then reverse. "
        "Trials contribute equally within participant. CNN seed metrics are averaged before participants are weighted equally. "
        "The percentile intervals resample participants and are exploratory.", "",
        "| Decoder | P + stage MSE | Add dynamics MSE | Change | Exploratory 95% interval | Improved |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model,label in LABELS.items():
        d=summary["contrast"][model]["82"]
        lo,hi=d["delta_ci"]
        report.append(f"| {label} | {d['baseline']:.6f} | {d['full']:.6f} | {d['delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {d['improved']}/12 |")
    report += ["", "The predictors of error are observed wrist position and stage, with optional velocity, acceleration and jerk. "
               "The primary 82 ms contrast adds velocity and acceleration. The stage plots are descriptive within-stage rank correlations. "
               "The speed table uses medians across participant estimates, with at least five trials per group in every fit.",
               "", "![Smoothing checks](smoothing_comparison.png)", "", "![Stage associations](stage_speed_associations.png)", "",
               "[Participant error models](participant_error_models.csv) · [Streak speeds](participant_streak_speeds.csv) · "
               "[Stage correlations](participant_stage_correlations.csv) · [Summary](population_summary.json)", ""]
    (out/"RESULTS.md").write_text("\n".join(report),encoding="utf-8")
    (out/"verification.json").write_text(json.dumps({"status":"passed","participants":12,"checked_alignment_series":24,
        "trial_targets":sum(r["trial_targets"] for r in alignment.values()),"trials":sum(r["contributing_trials"] for r in alignment.values()),
        "seed_level_error_model_fits":12*4*3*9*2,"participant_metric_sets":len(fits)},indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()

