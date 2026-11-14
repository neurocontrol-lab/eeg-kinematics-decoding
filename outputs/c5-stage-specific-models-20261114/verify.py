"""Independently check the C_5 participant-to-cohort rollup."""
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from math import isclose

out = Path(sys.argv[1])
def rows(name):
    with (out/name).open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))

saved = json.loads((out/"summary.json").read_text(encoding="utf-8"))
details = rows("participant_scores.csv")
common = saved["common_stages"]
for result in saved["balanced"]:
    actual = [r for r in details if r["policy"]=="Pure stage"
              and r["architecture"] == result["architecture"] and r["scope"] == result["scope"]
              and r["test_stage"] in common and r["train_stage"] in ("",r["test_stage"])]
    keys = {(r["participant"],r["test_stage"]) for r in actual}
    assert len(actual) == len(keys) == 12*len(common)
    for metric in ("mse","r2","r2_x","r2_y","r2_z"):
        assert result[metric+"_cells"] == len(actual)
        assert isclose(mean(float(r[metric]) for r in actual),result[metric],abs_tol=1e-12)
fits = rows("cnn_fit_manifest.csv")
assert len(fits)==saved["cnn_fits"]
for person in (f"P{i}" for i in range(1,13)):
    for family in ("cnn_elu","eegnet_regression","cnn_bilstm"):
        models=[r for r in fits if r["participant"]==person and r["architecture"]==family]
        assert sum(r["scope"]=="Pooled" for r in models)==1
        for stage in common:
            pair=[r for r in models if r["train_stage"]==stage]
            assert {r["scope"] for r in pair}=={"Specialist","Matched pooled"} and len(pair)==2
            assert pair[0]["n_train"]==pair[1]["n_train"]
result=dict(status="passed",balanced_rows=len(saved["balanced"]),matched_stages=len(common),
            participant_score_rows=len(details),gpu_fits=len(fits))
(out/"independent_verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result))
