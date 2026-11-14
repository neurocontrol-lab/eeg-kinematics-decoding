"""C_5: compare pooled and stage-specific ridge wrist decoders."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

root = Path.cwd()
sys.path.insert(0, str(root / "src"))
from eeg_kinematics.paths import RAW_DATA_DIR, REDUCED_DATA_DIR
from eeg_kinematics.timing_alignment import WINDOW, make_windows, window_starts
from eeg_kinematics.error_dynamics import derivatives
sys.path.insert(0, str(root / "outputs/t1-r6-followup-20261110"))
from r6_population_followup import raw_struct, stage_alignment

STAGES = ("Preparation", "Reach", "Preload", "Loading", "Lift/hold",
          "Lowering", "Release", "Return")
REFERENCE = root / "outputs/timing-alignment-population-gpu-20261028T133000Z"
POLICIES = ("Pure stage", "Target stage")
SEEDS = (42, 43, 44, 45, 46)
MIN_TRAIN_WINDOWS = 20
MIN_TRAIN_TRIALS = 8
MIN_TEST_TRIALS = 5


def write_csv(path, records):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temp.replace(path)


def extract(person, meta):
    eeg = loadmat(REDUCED_DATA_DIR / f"eeg_{person}.mat", variable_names=["eeg"])["eeg"].T.astype(np.float32)
    pos = loadmat(REDUCED_DATA_DIR / f"kin_{person}.mat", variable_names=["kin"])["kin"].astype(np.float32)
    bounds = {int(s): pair for s, pair in meta["series_bounds"].items()}
    starts = loadmat(REDUCED_DATA_DIR / f"session_start_{person}.mat")["session_start"].ravel().astype(int) - 1
    assert len(eeg) == len(pos) == bounds[9][1] and len(starts) == 9
    np.testing.assert_array_equal(starts, [bounds[i][0] for i in range(1, 10)])
    scaling = meta["scaling"]
    assert scaling["fit_series"] == list(range(1, 7))
    with np.load(REFERENCE / person / "split_indices.npz") as split:
        targets = np.concatenate([split[k+"_targets"] for k in ("train","test")])
        series = np.concatenate([split[k+"_series_ids"] for k in ("train","test")])
    series = series.astype(np.int64)
    assert set(series) == {1, 2, 3, 4, 5, 6, 8, 9}
    with np.load(REFERENCE / person / "ridge_gap1_predictions.npz") as saved:
        is_test = series >= 8
        np.testing.assert_array_equal(saved["target_sample_index"], targets[is_test])
        np.testing.assert_array_equal(saved["series_id"], series[is_test])
        expected = (pos[targets[is_test]] - scaling["target_mean"]) / scaling["target_scale"]
        np.testing.assert_allclose(saved["y_true"], expected, rtol=1e-5, atol=1e-6)
    all_lifts = raw_struct(RAW_DATA_DIR / person / f"{person}_AllLifts.mat", "P")
    cols = {str(name): i for i, name in enumerate(all_lifts.ColNames)}
    kept_stage, trial, pure, condition, speed = (np.zeros(len(targets), dtype=dtype)
                                              for dtype in (int, int, bool, int, float))
    checks = []
    for s in sorted(set(series)):
        lo, hi = bounds[s]
        stages, trials, check = stage_alignment(person, s, hi-lo, all_lifts.AllLifts, cols)
        mask = series == s
        t = targets[mask]-lo
        start = window_starts(targets[mask], 1)-lo
        assert np.all(start >= 0)
        q = trials[t]
        kept_stage[mask] = stages[t]
        trial[mask] = np.where(q > 0, 100*s+q, 0)
        pure[mask] = (q > 0) & (trials[start] == q) & (stages[start] == stages[t])
        rows = all_lifts.AllLifts[all_lifts.AllLifts[:,cols["Run"]] == s]
        condition[mask] = np.where(q > 0, rows[q-1,cols["CurW"]]*10+rows[q-1,cols["CurS"]], 0)
        speed[mask] = np.linalg.norm(derivatives(pos[lo:hi], 41)[t,0],axis=1)
        checks.append({"series": int(s), **check})
    included = trial > 0
    windows = make_windows(eeg,targets[included],series[included],bounds,1)
    windows -= np.asarray(scaling["eeg_mean"],dtype=np.float32)
    windows /= np.asarray(scaling["eeg_scale"],dtype=np.float32)
    y = ((pos[targets[included]] - scaling["target_mean"]) / scaling["target_scale"]).astype(np.float32)
    return (windows.reshape(-1,WINDOW*32), y, series[included], kept_stage[included],
            trial[included], pure[included], condition[included], speed[included], checks)


def class_matched_indices(pool, specialist, strata, seed):
    rng = np.random.default_rng(seed)
    matched = []
    for key,n in sorted(Counter(strata[specialist]).items()):
        candidates = pool[strata[pool] == key]
        matched.extend(rng.choice(candidates, size=n, replace=False))
    result = np.array(sorted(matched))
    assert Counter(strata[result]) == Counter(strata[specialist])
    assert len(result) == len(np.unique(result)) == len(specialist)
    return result


def run_person(person, meta, out):
    x,y,series,stage,trial,pure,cond,speed,checks = extract(person, meta)
    alpha=float(meta["gaps"]["1"]["ridge"]["selected_alpha"])
    strata=series*100+cond
    coverage, metrics, fits, support = [], [], [], []
    for policy in POLICIES:
        eligible = pure if policy == "Pure stage" else np.ones_like(pure)
        train=np.flatnonzero(eligible & (series<=6))
        test=np.flatnonzero(eligible & (series>=8))
        test_trials=trial[test]
        truth=y[test].astype(float)
        test_x=x[test]
        trial_index = {}
        for stage_id, name in enumerate(STAGES,start=1):
            tr=train[stage[train]==stage_id]
            te=np.flatnonzero(stage[test]==stage_id)
            ntrain=len(np.unique(trial[tr]))
            ntest=len(np.unique(test_trials[te]))
            coverage.append(dict(participant=person,policy=policy,stage=name,train_windows=len(tr),train_trials=ntrain,
                                 test_windows=len(te),test_trials=ntest,median_speed=float(np.median(speed[test[te]])) if len(te) else None,
                                 target_variance=float(np.mean(truth[te].var(axis=0))) if len(te) else None,
                                 train_conditions=";".join(f"{c}:{n}" for c,n in sorted(Counter(cond[tr]).items())),
                                 test_conditions=";".join(f"{c}:{n}" for c,n in sorted(Counter(cond[test[te]]).items()))))
            if ntest>=MIN_TEST_TRIALS:
                trial_index[stage_id]=[te[test_trials[te]==tid] for tid in np.unique(test_trials[te])]
            if len(tr)>=MIN_TRAIN_WINDOWS and ntrain>=MIN_TRAIN_TRIALS:
                support.append((policy,stage_id,tr))
        def ridge(indices, model, seed):
            estimator=Ridge(alpha=alpha,solver="lsqr",tol=1e-3,max_iter=300).fit(x[indices],y[indices])
            fits.append(dict(participant=person,policy=policy,model=model,seed=seed,windows=len(indices),
                             trials=len(np.unique(trial[indices])),alpha=alpha,max_iter_used=int(max(estimator.n_iter_)),
                             train_stage_count=len(np.unique(stage[indices]))))
            return estimator.predict(test_x).astype(float)
        def measure(pred,model,train_stage="",seed=""):
            error = (pred-truth)**2
            for stage_id, indices in trial_index.items():
                trial_errors = np.array([error[idx].mean(axis=0) for idx in indices])
                vals = trial_errors.mean(axis=0)
                metrics.append(dict(participant=person,policy=policy,model=model,train_stage=train_stage,
                                    test_stage=STAGES[stage_id-1],seed=seed,trials=len(indices),
                                    windows=sum(map(len,indices)),mse=float(vals.mean()),
                                    mse_x=vals[0],mse_y=vals[1],mse_z=vals[2]))
        measure(ridge(train,"Pooled ridge",""),"Pooled ridge")
        measure(y[train].mean(axis=0),"Pooled mean")
        for name,stage_id,tr in support:
            if name!=policy:
                continue
            stage_name=STAGES[stage_id-1]
            measure(ridge(tr,stage_name,""),"Specialist ridge",stage_name)
            measure(y[tr].mean(axis=0),"Stage mean",stage_name)
            for seed in SEEDS:
                matched=class_matched_indices(train,tr,strata,seed)
                measure(ridge(matched,stage_name+" matched",seed),"Matched pooled ridge",stage_name,seed)
    write_csv(out/f"{person}-coverage.csv",coverage)
    write_csv(out/f"{person}-fits.csv",fits)
    (out/f"{person}-clocks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    write_csv(out/f"{person}-metrics.csv",metrics)
    print("COMPLETE", person, len(fits), flush=True)


def run(out):
    for name in ("verification.json","population_verification.json"):
        assert json.loads((REFERENCE/name).read_text())["status"]=="passed"
    reference=json.loads((REFERENCE/"results.json").read_text(encoding="utf-8"))
    with threadpool_limits(3):
        for n in range(1,13):
            person=f"P{n}"
            if not (out/f"{person}-metrics.csv").exists():
                run_person(person,reference["participant_results"][person],out)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
