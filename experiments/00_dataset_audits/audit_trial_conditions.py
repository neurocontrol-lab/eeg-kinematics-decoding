"""C1: audit named trial conditions and coverage without training a model."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
_repo = next(p for p in Path(__file__).resolve().parents if (p / "src/eeg_kinematics/paths.py").is_file())
sys.path.insert(0, str(_repo / "src"))
from eeg_kinematics.paths import REPO_ROOT, RAW_DATA_DIR, REDUCED_DATA_DIR, resolve_recorded_source

import numpy as np
from scipy.io import loadmat, whosmat

ROOT = REPO_ROOT
TYPES = {1: "weight", 2: "friction", 3: "mixed"}
WEIGHTS = {1: 165, 2: 330, 4: 660}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(number):
    path = RAW_DATA_DIR / f"P{number}/P{number}_AllLifts.mat"
    p = loadmat(path, squeeze_me=True, struct_as_record=False)["P"]
    cols = {str(name): i for i, name in enumerate(p.ColNames)}
    assert all(k in cols for k in ["Run", "Lift", "CurW", "CurS", "PrevW", "PrevS", "BlockType", "StartTime", "LEDOn", "tHandStop"])
    assert set(p.AllLifts[:, cols["Run"]].astype(int)) == set(range(1, 10))
    return p.AllLifts, cols, path


def surface_mapping():
    """Use local WS string labels, rather than infer labels from numeric codes."""
    labels = {}
    checked = 0
    a, cols, _ = metadata(1)
    for run in range(1, 10):
        w = loadmat(RAW_DATA_DIR / f"P1/WS_P1_S{run}.mat", squeeze_me=True, struct_as_record=False)["ws"]
        rows = a[a[:, cols["Run"]] == run]
        assert len(rows) == len(w.win)
        for q, row in zip(w.win, rows):
            code, label = int(q.surf), str(q.surf_id)
            assert int(row[cols["CurS"]]) == code
            assert int(q.weight) == int(row[cols["CurW"]])
            assert str(q.weight_id).replace(" ", "") == f"{WEIGHTS[int(q.weight)]}g"
            if code in labels:
                assert labels[code] == label
            labels[code] = label
            checked += 1
    assert set(labels) == {1, 2, 3}
    return labels, checked


def nullable(value):
    return int(value) if np.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (args.output_dir or ROOT / "outputs" / f"condition-audit-{stamp}").resolve()
    assert out.parent == (ROOT / "outputs").resolve()
    out.mkdir(exist_ok=False)
    labels, ws_checked = surface_mapping()
    report = {"status": "running", "scope": "C1 trial-condition audit, P1–P12", "participants": {},
              "metadata_columns": {"series_type": "BlockType", "weight": "CurW", "surface": "CurS",
                                   "previous_weight": "PrevW", "previous_surface": "PrevS"},
              "weight_grams": WEIGHTS, "local_WS_surface_labels": labels,
              "P1_WS_trials_checked": ws_checked,
              "label_note": "Surface strings are verified in all P1 WS trials. Other participants' condition audit retains original codes; labels use that shared local mapping.",
              "experiment_sha256": sha(Path(__file__)), "source_sha256": {}}
    records = []
    for number in range(1, 13):
        name = f"P{number}"
        a, cols, path = metadata(number)
        report["source_sha256"][f"../{name}/{path.name}"] = sha(path)
        starts = loadmat(REDUCED_DATA_DIR / f"session_start_{name}.mat")["session_start"].ravel().astype(int) - 1
        shape_eeg = next(shape for var, shape, _ in whosmat(REDUCED_DATA_DIR / f"eeg_{name}.mat") if var == "eeg")
        shape_kin = next(shape for var, shape, _ in whosmat(REDUCED_DATA_DIR / f"kin_{name}.mat") if var == "kin")
        assert shape_eeg == (32, shape_kin[0]) and shape_kin[1] == 3
        assert len(starts) == 9 and starts[0] == 0 and np.all(np.diff(starts) > 128)
        runs, split_counts, split_conditions = {}, {}, {}
        for run in range(1, 10):
            rows = a[a[:, cols["Run"]] == run]
            types = sorted({int(x) for x in rows[:, cols["BlockType"]]})
            assert len(types) == 1 and types[0] in TYPES
            conditions = Counter((int(x[cols["CurW"]]), int(x[cols["CurS"]])) for x in rows)
            runs[str(run)] = {"series_type": TYPES[types[0]], "block_type": types[0], "n_trials": len(rows),
                              "weights": sorted({int(x) for x in rows[:, cols["CurW"]]}),
                              "surfaces": sorted({int(x) for x in rows[:, cols["CurS"]]}),
                              "conditions": {f"w{w}_s{s}": n for (w, s), n in sorted(conditions.items())}}
            for row in rows:
                weight, surface = int(row[cols["CurW"]]), int(row[cols["CurS"]])
                pw, ps = nullable(row[cols["PrevW"]]), nullable(row[cols["PrevS"]])
                records.append({"participant": name, "run": run, "trial": int(row[cols["Lift"]]),
                    "v2_split": "train" if run <= 6 else "validation" if run == 7 else "test",
                    "block_type": types[0], "series_type": TYPES[types[0]], "weight_code": weight,
                    "weight_grams": WEIGHTS[weight], "surface_code": surface, "surface_label": labels[surface],
                    "previous_weight_code": pw, "previous_surface_code": ps,
                    "weight_changed": None if pw is None else weight != pw,
                    "surface_changed": None if ps is None else surface != ps,
                    "metadata_start_time": float(row[cols["StartTime"]])})
        for label, ids in [("train", range(1, 7)), ("validation", [7]), ("test", [8, 9])]:
            split_counts[label] = sum(runs[str(run)]["n_trials"] for run in ids)
            split_conditions[label] = sorted({c for run in ids for c in runs[str(run)]["conditions"]})
        missing = {s: sorted(set(split_conditions[s]) - set(split_conditions["train"])) for s in ["validation", "test"]}
        report["participants"][name] = {"runs": runs, "split_trial_counts": split_counts,
            "joint_conditions_by_v2_split": split_conditions, "heldout_conditions_absent_from_v2_train": missing}
        print("AUDITED", name, "types", [runs[str(r)]["series_type"] for r in range(1, 10)], "missing", missing, flush=True)
    assert all(report["participants"][f"P{p}"]["runs"][str(r)]["block_type"] == 1 for p in range(1, 13) for r in [7, 8, 9])
    p1 = report["participants"]["P1"]["runs"]
    for run in [1, 4, 7, 8, 9]:
        assert p1[str(run)]["block_type"] == 1 and p1[str(run)]["weights"] == [1, 2, 4]
        assert p1[str(run)]["surfaces"] == [3] and labels[3] == "sandpaper"
    report["C2_P1_plan"] = {"train_runs": [1, 4], "validation_runs": [7], "test_runs": [8, 9],
        "classes_grams": [165, 330, 660], "fixed_surface": "sandpaper", "surface_code": 3,
        "all_classes_present_in_each_selected_run": True, "selected_trial_counts": {"train": 68, "validation": 34, "test": 68},
        "scope": "Within-P1 weight discrimination at a fixed surface, not surface classification or participant transfer"}
    report["source_files_unchanged"] = all(sha(resolve_recorded_source(p)) == h for p, h in report["source_sha256"].items())
    assert report["source_files_unchanged"]
    with (out / "trial_conditions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    report["trial_rows"] = len(records)
    report["status"] = "complete"
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    lines = ["# C1: trial-condition audit", "", "P1–P12 metadata, with named columns and trial-level current/previous conditions.",
             "All P1 WS trials verify physical weight strings and surface-code labels. Numeric surface codes are not series types.", "",
             "| Participant | Run types 1–6 | Runs 7–9 | V2 held-out joint conditions absent from training |", "|---|---|---|---|"]
    for label, item in report["participants"].items():
        lines.append(f"| {label} | " + ", ".join(item["runs"][str(r)]["series_type"] for r in range(1, 7)) +
                     " | Weight, fixed sandpaper | " + str(item["heldout_conditions_absent_from_v2_train"]) + " |")
    lines += ["", "## C2 setup supported by the audit", "",
        "P1 weight runs 1 and 4 train (68 trials), run 7 validates (34), runs 8–9 test (68). Every selected run contains all three weights at fixed sandpaper.",
        "Weight varies between trials within these runs. Previous conditions and repeated/changed flags are retained in trial_conditions.csv.",
        "The v2 training split mixes series types while held-out runs 7–9 are weight series. Matching joint conditions alone does not make their task contexts identical.",
        "All trial conditions and per-run counts are in results.json and trial_conditions.csv. Existing model artifacts are untouched."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
