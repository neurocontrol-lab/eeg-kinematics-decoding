"""Independently verify C1 metadata and C2 saved predictions, without refitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
_repo = next(p for p in Path(__file__).resolve().parents if (p / "src/eeg_kinematics/paths.py").is_file())
sys.path.insert(0, str(_repo / "src"))
from eeg_kinematics.paths import REPO_ROOT, RAW_DATA_DIR, REDUCED_DATA_DIR

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = REPO_ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_metrics(y, probability, expected):
    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-5)
    prediction = probability.argmax(axis=1)
    for key, value in [("accuracy", accuracy_score(y, prediction)),
                       ("balanced_accuracy", balanced_accuracy_score(y, prediction)),
                       ("macro_f1", f1_score(y, prediction, average="macro", zero_division=0))]:
        np.testing.assert_allclose(value, expected[key], rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(confusion_matrix(y, prediction, labels=[0, 1, 2]), expected["confusion_matrix"])
    assert len(y) == expected["n"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report-dir", type=Path, help="Write verification here instead of changing the saved run.")
    args = parser.parse_args()
    report_dir = (args.report_dir or args.output_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir.resolve()
    assert out.parent == (ROOT / "outputs").resolve()
    d = json.loads((out / "results.json").read_text())
    assert d["status"] == "complete" and d["source_files_unchanged"]
    assert sha(out / "experiment_source.py") == d["source_sha256"]["experiment"]
    assert sha(out / "declared_protocol.md") == d["source_sha256"]["declared_protocol"]
    audit_dir = ROOT / d["audit_directory"]
    assert sha(audit_dir / "results.json") == d["source_sha256"]["condition_audit"]
    audit = json.loads((audit_dir / "results.json").read_text())
    assert audit["status"] == "complete" and audit["source_files_unchanged"]
    assert sha(audit_dir / "analysis_source.py") == audit["experiment_sha256"]
    metadata_rows = list(csv.DictReader((audit_dir / "trial_conditions.csv").open()))
    assert len(metadata_rows) == audit["trial_rows"]
    audited = 0
    for number in range(1, 13):
        label = f"P{number}"
        raw = RAW_DATA_DIR / label / f"{label}_AllLifts.mat"
        assert sha(raw) == audit["source_sha256"][f"../{label}/{raw.name}"]
        p = loadmat(raw, squeeze_me=True, struct_as_record=False)["P"]
        cols = {str(n): i for i, n in enumerate(p.ColNames)}
        rows = [r for r in metadata_rows if r["participant"] == label]
        assert len(rows) == len(p.AllLifts)
        for saved, original in zip(rows, p.AllLifts):
            for saved_name, column in [("run", "Run"), ("trial", "Lift"), ("weight_code", "CurW"), ("surface_code", "CurS"), ("block_type", "BlockType")]:
                assert int(saved[saved_name]) == int(original[cols[column]])
            for saved_name, column in [("previous_weight_code", "PrevW"), ("previous_surface_code", "PrevS")]:
                assert (None if saved[saved_name] == "" else int(saved[saved_name])) == (None if not np.isfinite(original[cols[column]]) else int(original[cols[column]]))
            audited += 1
    trials = json.loads((out / "trial_manifest.json").read_text())
    assert len(trials) == 170 and len({r["key"] for r in trials}) == 170
    assert d["runs"] == {"train": [1, 4], "validation": [7], "test": [8, 9]}
    assert sum(v["all_trial_EEG_and_wrist_checks"] for v in d["clock_checks"].values()) == 170
    for run in [1, 4, 7, 8, 9]:
        rr = [r for r in trials if r["run"] == run]
        assert len(rr) == 34 and {r["class_index"] for r in rr} == {0, 1, 2}
        assert all(a["end_index"] <= b["start_index"] for a, b in zip(rr, rr[1:]))
    # Recompute scaler statistics independently from training intervals.
    eeg = np.asarray(loadmat(REDUCED_DATA_DIR / 'eeg_P1.mat', variable_names=["eeg"])["eeg"].T, dtype=np.float32, order="C")
    scaler = StandardScaler()
    for row in trials:
        if row["split"] == "train":
            scaler.partial_fit(eeg[row["start_index"]:row["end_index"]])
    np.testing.assert_allclose(scaler.mean_, d["scaling"]["eeg_mean"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scaler.scale_, d["scaling"]["eeg_scale"], rtol=1e-12, atol=1e-12)
    assert scaler.n_samples_seen_ == d["scaling"]["n_samples"]
    del eeg
    checked_files, metric_checks, permutation_checks = 0, 0, 0
    for name, entry in d["models"].items():
        for split in ["validation", "test"]:
            rr = [r for r in trials if r["split"] == split]
            with np.load(out / f"{name}_{split}_predictions.npz") as z:
                keys, probability, y = z["trial_key"], z["window_prob"], z["window_y"]
                for row in rr:
                    mask = keys == row["key"]
                    np.testing.assert_array_equal(z["window_start"][mask], np.arange(row["start_index"], row["end_index"] - 128 + 1, 64))
                    assert np.all(y[mask] == row["class_index"])
                recomputed = np.stack([probability[keys == r["key"]].mean(axis=0) for r in rr])
                np.testing.assert_allclose(recomputed, z["trial_prob"], rtol=1e-10, atol=1e-12)
                np.testing.assert_array_equal(z["trial_y"], [r["class_index"] for r in rr])
                verify_metrics(y, probability, entry[split]["window_secondary"])
                verify_metrics(z["trial_y"], recomputed, entry[split]["trial"])
                metric_checks += 2
                for run in np.unique(z["trial_run"]):
                    mask = z["trial_run"] == run
                    verify_metrics(z["trial_y"][mask], recomputed[mask], entry[split]["by_run"][str(run)])
                    metric_checks += 1
                if split == "test" and name != "training_majority":
                    labels, run_ids = z["trial_y"], z["trial_run"]
                    pred = recomputed.argmax(axis=1)
                    expected_null = []
                    for a in range(34):
                        for b in range(34):
                            shifted = labels.copy()
                            for run, offset in [(8, a), (9, b)]:
                                mask = run_ids == run; shifted[mask] = np.roll(labels[mask], offset)
                            expected_null.append(balanced_accuracy_score(shifted, pred))
                    with np.load(out / f"{name}_circular_reference.npz") as ref:
                        np.testing.assert_allclose(expected_null, ref["balanced_accuracy"], atol=1e-12)
                    observed = entry[split]["trial"]["balanced_accuracy"]
                    p = float(np.mean(np.array(expected_null) >= observed - 1e-12))
                    np.testing.assert_allclose(p, entry[split]["circular_shift_reference"]["upper_tail_fraction"], atol=1e-12)
                    permutation_checks += len(expected_null)
            checked_files += 1
    logistic = d["models"]["logistic"]
    selected = max(logistic["candidates"], key=lambda c: c["validation_trials"]["balanced_accuracy"])
    assert selected["C"] == logistic["selected_C"]
    summary = {"status": "passed", "audit_trial_rows_checked": audited, "classifier_trials": 170,
               "source_and_protocol_hashes_match": True, "training_only_scaler_recomputed": True,
               "trial_window_boundaries_and_labels_checked": True, "saved_prediction_files_checked": checked_files,
               "metric_subsets_checked": metric_checks, "circular_offsets_recomputed": permutation_checks,
               "validation_only_logistic_selection_checked": True, "GPU_inference_probe": d["gpu_probe_device"]}
    (report_dir / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
