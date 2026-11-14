"""Independently rebuild the C_5 paired contrasts from saved metrics."""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

out = Path(sys.argv[1])
summary = json.loads((out/"population_summary.json").read_text(encoding="utf-8"))
with (out/"metrics.csv").open(encoding="utf-8",newline="") as f:
    rows = list(csv.DictReader(f))
models = defaultdict(dict)
for row in rows:
    group = row["participant"],row["policy"],row["test_stage"]
    key = row["model"],row["train_stage"],row["seed"]
    assert key not in models[group]
    models[group][key] = float(row["mse"])
deltas = defaultdict(list)
overall = defaultdict(lambda: defaultdict(list))
off_diagonal = defaultdict(lambda: defaultdict(list))
for (p,policy,test), m in models.items():
    pooled = m["Pooled ridge","",""]
    for (model,train,seed), specialist in m.items():
        if model!="Specialist ridge":
            continue
        deltas[policy,train,test,"Pooled ridge"].append(specialist-pooled)
        if train!=test:
            if ("Specialist ridge",test,"") in m:
                off_diagonal[policy][p].append(specialist-m["Specialist ridge",test,""])
            continue
        matched = [m["Matched pooled ridge",train,str(s)] for s in range(42,47)]
        for label,ref in (("Pooled ridge",pooled),("Matched pooled ridge",mean(matched)),("Stage mean",m["Stage mean",train,""])):
            diff=specialist-ref
            overall[policy,label][p].append(diff)
            if label!="Pooled ridge":
                deltas[policy,train,test,label].append(diff)

checks=0
for policy,entries in summary["cross_stage"].items():
    for entry in entries:
        values=deltas[policy,entry["train_stage"],entry["test_stage"],"Pooled ridge"]
        assert len(values)==entry["n"] and math.isclose(mean(values),entry["mean"],abs_tol=1e-12)
        checks+=1
for policy,entries in summary["diagonal"].items():
    for stage,models in entries.items():
        for label,entry in models.items():
            values=deltas[policy,stage,stage,label]
            assert len(values)==entry["n"] and math.isclose(mean(values),entry["mean"],abs_tol=1e-12)
            checks+=1
for (policy,label), values in overall.items():
    entry=summary["overall"][policy][label]
    assert len(values)==entry["n"] and math.isclose(mean(map(mean,values.values())),entry["mean"],abs_tol=1e-12)
    checks+=1
cross={policy:{"participants":len(vals),"comparison_cells":sum(map(len,vals.values())),
               "cross_minus_same_mse":mean(map(mean,vals.values()))} for policy,vals in off_diagonal.items()}
result={"status":"passed","participant_policy_test_groups":len({(r["participant"],r["policy"],r["test_stage"]) for r in rows}),
        "metric_rows":len(rows),"independently_recomputed_aggregates":checks,"cross_stage_vs_same_stage":cross}
(out/"independent_verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2))
