"""Verify C_5 fit coverage and summarize paired participant effects."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run import POLICIES, SEEDS, STAGES, write_csv

PEOPLE = [f"P{i}" for i in range(1,13)]
COMPARATORS = ("Pooled ridge", "Matched pooled ridge", "Stage mean")


def read(path, suffix):
    result = []
    for person in PEOPLE:
        with (path / f"{person}-{suffix}.csv").open(encoding="utf-8",newline="") as stream:
            result.extend(csv.DictReader(stream))
    return result


def interval(values):
    values = np.asarray(values)
    sample = np.random.default_rng(42).integers(len(values),size=(10000,len(values)))
    lo,hi = np.quantile(values[sample].mean(axis=1),[.025,.975])
    return dict(n=len(values),mean=float(values.mean()),low=float(lo),high=float(hi),better=int((values<0).sum()))


def rollup(out):
    scores = read(out,"metrics")
    fits = read(out,"fits")
    coverage = read(out,"coverage")
    clocks = [x for p in PEOPLE for x in json.loads((out/f"{p}-clocks.json").read_text(encoding="utf-8"))]
    assert len(coverage)==len(PEOPLE)*len(POLICIES)*len(STAGES)
    assert len(clocks) == 96
    support = {(r["participant"],r["policy"],r["stage"]):r for r in coverage}
    fit_index = {(r["participant"],r["policy"],r["model"],r["seed"]):r for r in fits}
    assert len(fit_index)==len(fits)
    assert max(int(r["max_iter_used"]) for r in fits)<300
    for (p,policy,s), row in support.items():
        if int(row["train_windows"])>=20 and int(row["train_trials"])>=8:
            specialist = fit_index[p,policy,s,""]
            assert int(specialist["windows"])==int(row["train_windows"])
            for seed in SEEDS:
                fit = fit_index[p,policy,s+" matched",str(seed)]
                assert fit["windows"]==specialist["windows"] and fit["alpha"]==specialist["alpha"]
    indexed={}
    base_targets={}
    for row in scores:
        p,policy,test,stage,model,seed=(row[k] for k in ("participant","policy","test_stage","train_stage","model","seed"))
        key=p,policy,test,model,stage,seed
        assert key not in indexed
        indexed[key]=float(row["mse"])
        assert np.isclose(mean(float(row[k]) for k in ("mse_x","mse_y","mse_z")),indexed[key],rtol=1e-12)
        target_key=p,policy,test
        targets=(row["trials"],row["windows"])
        assert targets==(support[target_key]["test_trials"],support[target_key]["test_windows"])
        assert base_targets.setdefault(target_key,targets)==targets

    deltas=[]
    diag={}
    matrices={}
    overall={}
    for policy in POLICIES:
        matrix=[]
        for tr in STAGES:
            for te in STAGES:
                values=[]
                for p in PEOPLE:
                    specialist=indexed.get((p,policy,te,"Specialist ridge",tr,""))
                    if specialist is None:
                        continue
                    pooled=indexed[p,policy,te,"Pooled ridge","",""]
                    values.append(specialist-pooled)
                    if tr==te:
                        controls=[pooled, mean(indexed[p,policy,te,"Matched pooled ridge",tr,str(seed)] for seed in SEEDS),
                                  indexed[p,policy,te,"Stage mean",tr,""]]
                        for label,reference in zip(COMPARATORS,controls):
                            deltas.append(dict(participant=p,policy=policy,stage=te,comparator=label,
                                               specialist_mse=specialist,reference_mse=reference,delta=specialist-reference))
                if values:
                    matrix.append(dict(train_stage=tr,test_stage=te,**interval(values)))
        matrices[policy]=matrix
        selected=[r for r in deltas if r["policy"]==policy]
        diag[policy]={}
        overall[policy]={}
        for label in COMPARATORS:
            participants=defaultdict(list)
            for t in STAGES:
                values=[r["delta"] for r in selected if r["comparator"]==label and r["stage"]==t]
                if values:
                    diag[policy].setdefault(t,{})[label]=interval(values)
            for row in selected:
                if row["comparator"]==label:
                    participants[row["participant"]].append(row["delta"])
            overall[policy][label]=interval([mean(v) for v in participants.values()])
    for name,rows in (("metrics",scores),("fit_manifest",fits),("stage_coverage",coverage),("paired_effects",deltas)):
        write_csv(out/f"{name}.csv",rows)
    verification=dict(status="passed",participants=len(PEOPLE),series_clocks=len(clocks),
                      trial_clocks=sum(x["clock_trials"] for x in clocks),
                      invalid_event_trials=sum(len(x["invalid_event_trials"]) for x in clocks),
                      fit_count=len(fits),metric_rows=len(scores),matched_control_fits=sum(r["seed"]!="" for r in fits))
    for filename,value in (("verification.json",verification),("population_summary.json",dict(
            diagonal=diag,cross_stage=matrices,overall=overall))):
        (out/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return diag,matrices,overall,verification


def plots(out,diag,matrices):
    stages=list(STAGES)
    fig,axes=plt.subplots(1,2,figsize=(14,6.6),layout="constrained")
    for ax,(policy,entries) in zip(axes,matrices.items()):
        grid=np.full((8,8),np.nan)
        counts=np.zeros((8,8),dtype=int)
        for row in entries:
            i,j=(stages.index(row[k]) for k in ("train_stage","test_stage"))
            grid[i,j],counts[i,j]=row["mean"],row["n"]
        lim=max(.1,float(np.nanquantile(abs(grid),.85)))
        im=ax.imshow(grid,cmap="RdBu_r",vmin=-lim,vmax=lim)
        for i,j in zip(*np.where(counts>0)):
            v=grid[i,j]
            ax.text(j,i,f"{v:+.2f}\nn={counts[i,j]}",ha="center",va="center",fontsize=8,
                    color="white" if abs(v)>.67*lim else "#173142",weight="bold" if i==j else "normal")
        ax.set(xticks=range(8),xticklabels=stages,yticks=range(8),yticklabels=stages,
               xlabel="Test stage",ylabel="Specialist training stage",title=policy)
        ax.tick_params(axis="x",rotation=38)
        fig.colorbar(im,ax=ax,shrink=.7,label="Specialist minus pooled MSE")
    fig.suptitle("Cross-stage wrist decoding with ridge",weight="bold")
    fig.savefig(out/"cross_stage_matrix.png",dpi=155)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12.5,5),layout="constrained",sharey=True)
    for ax,(policy,entries) in zip(axes,diag.items()):
        for k,(label,color) in enumerate(zip(COMPARATORS,("#0d7180","#6a5a97","#a34e24"))):
            pts=[(j,entries[s][label]) for j,s in enumerate(stages) if s in entries]
            y=[j+(k-1)*.19 for j,_ in pts]
            x=[v["mean"] for _,v in pts]
            ax.errorbar(x,y,xerr=([v["mean"]-v["low"] for _,v in pts],[v["high"]-v["mean"] for _,v in pts]),
                        fmt="o",ms=5,capsize=3,color=color,label=label)
        ax.axvline(0,color="#7e929d",ls="--")
        ax.set(yticks=range(8),yticklabels=stages,title=policy,xlabel="Specialist minus reference MSE")
        ax.grid(axis="y",alpha=.2)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False,fontsize=9)
    fig.suptitle("Same-stage specialists and matched controls",weight="bold")
    fig.savefig(out/"paired_stage_effects.png",dpi=155)
    plt.close(fig)


def report(out,diag,overall,verification):
    def fmt(v):
        return f'{v["mean"]:+.4f} [{v["low"]:+.4f}, {v["high"]:+.4f}]'
    lines=["# C_5 — Stage-specific versus pooled wrist-position decoding","",
           "P1–P12, ridge, fixed 2 ms prediction lead and 256 ms EEG windows. Models train on series 1–6 and score on series 8–9. Stages come from known event markers.","",
           "Values are specialist minus reference XYZ MSE. Negative values favour the specialist. Brackets are exploratory 95% intervals from resampling participants. The same test targets and trials are used within each comparison.",""]
    for policy,values in diag.items():
        lines.extend([f"## {policy}","","| Stage | Participants | Full pooled | Matched pooled | Stage mean |","| --- | ---: | ---: | ---: | ---: |"])
        for stage,row in values.items():
            lines.append(f'| {stage} | {row[COMPARATORS[0]]["n"]} | '+" | ".join(fmt(row[m]) for m in COMPARATORS)+" |")
        total=overall[policy]
        lines.extend(["",f"Across supported stages within participant, differences were {fmt(total[COMPARATORS[0]])} against full pooled ridge, "
                      f"{fmt(total[COMPARATORS[1]])} against matched pooled ridge and {fmt(total[COMPARATORS[2]])} against the stage mean.",""])
    lines.extend(["![Paired stage effects](paired_stage_effects.png)","","![Cross-stage matrix](cross_stage_matrix.png)","",
                  "Pure stage requires the full input window and target to fall in the same stage of the same trial. The target-stage sensitivity comparison retains preceding-phase context. Short stages can lack support under the purity rule. The cross-stage matrix reports separate improvement relative to the pooled model for each test stage. Each cell can contain a different number of participants.",
                  "","The matched pooled models have the same number of training windows and exact recording-series, weight and surface counts as the specialist. The pooled sampler may include samples from the specialist stage. Stage means use training targets only. Trial-balanced test scores give trials equal weight within each cell. Stage duration, kinematics and EEG preprocessing may contribute to differences. The stage coverage table records target variance, median speed and condition counts.",
                  "",f'Checked {verification["trial_clocks"]:,} trial clocks across {verification["series_clocks"]} series and {verification["fit_count"]:,} ridge fits. Excluded {verification["invalid_event_trials"]} trials with unusable event order or timestamps.',
                  "","[Protocol](PROTOCOL.md) · [Paired scores](paired_effects.csv) · [Cross-stage scores](metrics.csv) · [Coverage](stage_coverage.csv) · [Summary](population_summary.json) · [Verification](verification.json)",""])
    (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")

if __name__=="__main__":
    out=Path(sys.argv[1])
    diag,matrices,overall,verification=rollup(out)
    plots(out,diag,matrices)
    report(out,diag,overall,verification)
    print(json.dumps(dict(overall=overall,verification=verification),indent=2))
