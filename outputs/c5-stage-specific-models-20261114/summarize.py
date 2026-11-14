"""Aggregate trial-balanced C_5 scores with equal participant and stage weights."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PEOPLE = tuple(f"P{i}" for i in range(1,13))
ARCHITECTURES = ("ridge", "cnn_elu", "eegnet_regression", "cnn_bilstm")
LABELS = dict(zip(ARCHITECTURES, ("Ridge","CNN + ELU","EEGNet regression","CNN–BiLSTM")))
SCOPES = ("Specialist", "Pooled", "Matched pooled", "Stage mean", "Pooled mean")
STAGES = ("Preparation","Reach","Preload","Loading","Lift/hold","Lowering","Release","Return")


def read(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(row, metric):
    return float(row[metric]) if row[metric] != "" else None


def aggregate(out):
    scores, fits = [], []
    for person in PEOPLE:
        scores.extend(read(out/f"{person}-scores.csv"))
        fits.extend(read(out/f"{person}-fits.csv"))
    by_fit = {}
    for row in fits:
        key = tuple(row[k] for k in ("participant","architecture","scope","train_stage"))
        assert key not in by_fit and row["device"]=="GPU:0" and int(row["epochs"])==15
        by_fit[key] = row
    for (p,m,scope,stage), row in by_fit.items():
        if scope=="Matched pooled":
            assert row["n_train"]==by_fit[p,m,"Specialist",stage]["n_train"]
    indexed, target_sets = {}, {}
    for row in scores:
        p,policy,arch,scope,tr,te = (row[k] for k in ("participant","policy","architecture","scope","train_stage","test_stage"))
        key = p,policy,arch,scope,tr,te
        assert key not in indexed
        indexed[key]=row
        target = tuple(row[k] for k in ("windows","trials","variance_x","variance_y","variance_z"))
        assert target_sets.setdefault((p,policy,te),target)==target
        for a in "xyz":
            v=float(row[f"variance_{a}"])
            if v>1e-15:
                assert np.isclose(float(row[f"r2_{a}"]),1-float(row[f"mse_{a}"])/v)
            else:
                assert row[f"r2_{a}"]==""
    required = [(m,s) for m in ARCHITECTURES for s in SCOPES[:3]]+[("mean","Stage mean")]
    policy = "Pure stage"
    def diagonal(p,s,arch,scope):
        train_stage = "" if scope in ("Pooled","Pooled mean") else s
        return indexed.get((p,policy,arch,scope,train_stage,s))
    common = [s for s in STAGES if all(diagonal(p,s,a,k) for p in PEOPLE for a,k in required)]
    assert len(common) >= 5

    def summary(rows, kind, arch, stages):
        result = dict(scope=kind,architecture=arch,n_participants=len({r["participant"] for r in rows}),
                      n_stages=len(stages),n_cells=len(rows))
        for metric in ("mse","r2","r2_x","r2_y","r2_z"):
            values=defaultdict(list)
            for row in rows:
                val=score(row,metric)
                if val is not None:
                    values[row["participant"]].append(val)
            per_person=np.array([np.mean(v) for v in values.values()])
            result[metric]=float(per_person.mean()) if len(per_person) else None
            result[f"{metric}_cells"]=sum(map(len,values.values()))
        return result

    balanced = [summary([diagonal(p,s,a,k) for p in PEOPLE for s in common],k,a,common)
                for a,k in required+[("mean","Pooled mean")]]
    stage_rows=[]
    for policy_name in ("Pure stage","Target stage"):
        for a,k in required+[("mean","Pooled mean")]:
            for s in STAGES:
                tr = "" if k in ("Pooled","Pooled mean") else s
                rows=[indexed[p,policy_name,a,k,tr,s] for p in PEOPLE if (p,policy_name,a,k,tr,s) in indexed]
                if rows:
                    stage_rows.append(dict(policy=policy_name,stage=s,**summary(rows,k,a,[s])))

    matrix=[]
    for a in ARCHITECTURES:
        for tr in STAGES:
            for te in STAGES:
                rows=[indexed[p,policy,a,"Specialist",tr,te] for p in PEOPLE if (p,policy,a,"Specialist",tr,te) in indexed]
                if rows:
                    matrix.append(dict(train_stage=tr,test_stage=te,**summary(rows,"Specialist",a,[te])))

    paired=[]
    for a in ARCHITECTURES:
        for ref_arch,ref in ((a,"Pooled"),(a,"Matched pooled"),("mean","Stage mean")):
            for metric in ("mse","r2"):
                values=[]
                for p in PEOPLE:
                    pairs=[(score(diagonal(p,s,a,"Specialist"),metric),score(diagonal(p,s,ref_arch,ref),metric)) for s in common]
                    if all(x is not None and y is not None for x,y in pairs):
                        values.append(np.mean([x-y for x,y in pairs]))
                arr=np.array(values)
                if len(arr):
                    draws=np.random.default_rng(42).integers(0,len(arr),size=(10000,len(arr)))
                    low,high=np.quantile(arr[draws].mean(axis=1),[.025,.975])
                    paired.append(dict(architecture=a,reference=ref,metric=metric,n=len(arr),delta=float(arr.mean()),
                                       lower=float(low),upper=float(high)))
    for name,payload in (("participant_scores.csv",scores),("cnn_fit_manifest.csv",fits),("stage_scores.csv",stage_rows),
                         ("balanced_scores.csv",balanced),("cross_stage_scores.csv",matrix),("paired_effects.csv",paired)):
        write(out/name,payload)
    data=dict(status="verified",participants=len(PEOPLE),cnn_fits=len(fits),metric_rows=len(scores),
              common_stages=common,balanced=balanced,paired=paired)
    (out/"summary.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return data


def report(out,data):
    lines=["# C_5 — Stage-specific and pooled wrist decoders","",
           "Ridge and three CNN architectures. Pure-stage windows, 2 ms lead, held-out series 8–9. "
           "The balanced scores use the same participants, stages and test targets across models.","",
           "Common stages: "+", ".join(data["common_stages"])+".","",
           "First score each stage with equal trial weight, then average stages within participant and participants equally. "
           "R² uses the held-out trial-weighted mean for each axis as its reference. Positive R² beats that reference. "
           "MSE uses training-standardized wrist coordinates.",""]
    for title, scopes in (("Stage-specific decoders",("Specialist","Stage mean")),
                          ("Pooled decoders",("Pooled","Pooled mean")),
                          ("Sample-matched pooled controls",("Matched pooled",))):
        lines += ["## "+title,"","| Decoder | Balanced MSE ↓ | Mean XYZ R² ↑ | R² X | R² Y | R² Z |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in data["balanced"]:
            if row["scope"] in scopes:
                label=LABELS.get(row["architecture"],row["scope"])
                entries=[f"{row[m]:.4f}" if row[m] is not None else "—" for m in ("mse","r2","r2_x","r2_y","r2_z")]
                lines.append("| "+" | ".join([label,*entries])+" |")
        lines.append("")
    lines += ["The CNNs use seed 42 and 15 epochs. Stage R² is sensitive to low within-stage target variance. "
              "The strict comparison excludes stages without support for all models and participants. "
              "The stage scores include available participants and show support counts. Ridge also retains its target-stage sensitivity results.","",
              f"Verified {data['cnn_fits']} CNN fits and {data['metric_rows']} score rows.","",
              "[Protocol](PROTOCOL.md) · [Balanced scores](balanced_scores.csv) · [Stage scores](stage_scores.csv) · "
              "[Cross-stage specialist scores](cross_stage_scores.csv) · [Paired effects](paired_effects.csv) · "
              "[Participant scores](participant_scores.csv) · [Fit manifest](cnn_fit_manifest.csv)",""]
    (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    out = Path(sys.argv[1])
    data = aggregate(out)
    report(out,data)
    print(json.dumps(data,indent=2))
