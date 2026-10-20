"""Evaluate six saved P1 models without fitting on any target participant.

Primary comparison: runs 8–9 of P2–P12, matching the source P1 test-run IDs.
Secondary: all nine target runs, reported individually and pooled within subject.
Raw recordings verify wrist provenance; inference uses the original reduced files.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
_repo = next(p for p in Path(__file__).resolve().parents if (p / "src/eeg_kinematics/paths.py").is_file())
sys.path.insert(0, str(_repo / "src"))
from eeg_kinematics.paths import REPO_ROOT, REDUCED_DATA_DIR

import numpy as np
import scipy
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import matplotlib.pyplot as plt

REPO = REPO_ROOT
NAMES = ("training_mean", "ridge", "cnn", "cnn_bilstm", "cnn_elu", "eegnet_regression")
NEURAL = NAMES[2:]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(y, p):
    assert y.shape == p.shape and np.isfinite(y).all() and np.isfinite(p).all()
    return {"n_windows": len(y),
            "mse": mean_squared_error(y, p, multioutput="raw_values").tolist(),
            "mae": mean_absolute_error(y, p, multioutput="raw_values").tolist(),
            "r2": r2_score(y, p, multioutput="raw_values").tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO / "outputs/six-model-comparison-p1-gpu-20260930T095506Z")
    args = parser.parse_args()
    source = args.source.resolve()
    original = json.loads((source / "results.json").read_text())
    assert original["status"] == "complete" and original["subject"] == "P1"
    assert original["window_samples"] == 128 and original["stride_samples"] == 64
    assert original["scaling"]["fit_runs"] == [1, 2, 3, 4, 5, 6]
    artifacts = [source / "results.json", source / "training_mean_weights.npz", source / "ridge_weights.npz"]
    artifacts += [source / f"{name}.keras" for name in NEURAL]
    hashes = {f.name: digest(f) for f in artifacts}
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("A GPU is required; no silent CPU fallback.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    with tf.device("/GPU:0"):
        probe = tf.linalg.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
    assert "GPU:0" in probe.device
    print("GPU verified", probe.device, flush=True)
    models = {n: tf.keras.models.load_model(source / f"{n}.keras", compile=False) for n in NEURAL}
    for model in models.values():
        model.trainable = False
    initial_weights = {n: [w.copy() for w in m.get_weights()] for n, m in models.items()}
    mean = np.load(source / "training_mean_weights.npz")["mean"]
    with np.load(source / "ridge_weights.npz") as saved:
        coef, intercept = saved["coef"], saved["intercept"]
    scale = original["scaling"]
    # Reproduce float32 StandardScaler's separate in-place subtract/divide steps.
    # These constants come from P1 training samples; no target fitting occurs.
    em, es = np.array(scale["eeg_mean"]), np.array(scale["eeg_scale"])
    km, ks = np.array(scale["target_mean"]), np.array(scale["target_scale"])
    assert np.all(es > 0) and np.all(ks > 0)
    spec = importlib.util.spec_from_file_location("provenance", REPO / "experiments/00_dataset_audits/verify_kinematics.py")
    provenance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provenance)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / "outputs" / f"frozen-p1-cross-participant-gpu-{stamp}"
    out.mkdir(exist_ok=False)
    (out / "experiment_source.py").write_bytes(Path(__file__).read_bytes())
    result = {
        "status": "running", "experiment": "P1-trained, zero-adaptation cross-participant evaluation",
        "source_directory": source.relative_to(REPO).as_posix(), "source_artifact_sha256": hashes,
        "experiment_sha256": digest(Path(__file__)), "scaling": scale,
        "frozen": {"models": True, "P1_scalers": True, "target_fit_calls": 0,
                   "batch_normalization": "inference statistics; model.trainable=False"},
        "source_training_runs": [1, 2, 3, 4, 5, 6], "source_validation_runs": [7],
        "source_control_runs": [8, 9], "target_participants": list(range(2, 13)),
        "primary_target_runs": [8, 9], "secondary_target_runs": list(range(1, 10)),
        "window_samples": 128, "stride_samples": 64, "target_lead_after_last_input_samples": 1,
        "target_axes": ["Wrist X", "Wrist Y", "Wrist Z"],
        "metric_units": "P1-training-standardized reduced targets, not physical distances",
        "gpu": tf.config.experimental.get_device_details(gpus[0]), "gpu_probe_device": probe.device,
        "P1_control_tolerances": {"prediction_rtol": 1e-3, "prediction_atol": 1e-2,
            "prediction_mean_absolute_difference_max": 1e-3, "metric_rtol": 1e-3, "metric_atol": 1e-4,
            "reason": "Cross-process GPU LSTM replay differed by max 0.005582 and mean 0.000056 on bitwise-identical notebook inputs, for both trainable settings; aggregate metrics remain tightly checked."},
        "versions": {"python": sys.version.split()[0], "tensorflow": tf.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "limits": ["One source participant and one trained seed; this is not population/LOPO evaluation.",
                   "Target-participant scores are evaluation only, not model selection or adaptation.",
                   "All-run target scores and P1 held-out-run scores measure different run compositions.",
                   "Run IDs alone do not ensure matched friction; report each participant's metadata.",
                   "Existing reduced EEG and targets have undocumented upstream processing; target values are range-scaled per participant.",
                   "Frozen downstream scalers do not remove upstream participant-wide normalization or recover physical units.",
                   "Overlapping windows are correlated; participant macro summaries are descriptive, not independent-window confidence intervals."],
        "participants": {},
    }

    def checkpoint():
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    checkpoint()
    print("OUTPUT_DIRECTORY", out, flush=True)
    started = time.perf_counter()
    for participant in range(1, 13):
        label = f"P{participant}"
        print("VERIFY_RAW", label, flush=True)
        prov = provenance.report_subject(participant)
        for axis in prov["axes"].values():
            correlations = axis["correlation_with_original_sensors_1_to_4"]
            assert correlations[3] > .999 and correlations[3] == max(correlations), (label, correlations)
        data = REDUCED_DATA_DIR
        eeg_path, kin_path = data / f"eeg_{label}.mat", data / f"kin_{label}.mat"
        eeg = np.asarray(loadmat(eeg_path, variable_names=["eeg"])["eeg"].T, dtype=np.float32, order="C")
        kin = np.asarray(loadmat(kin_path, variable_names=["kin"])["kin"], dtype=np.float32, order="C")
        starts = loadmat(data / f"session_start_{label}.mat")["session_start"].ravel().astype(np.int64) - 1
        assert eeg.shape == (len(kin), 32) and kin.shape[1] == 3
        assert len(starts) == 9 and starts[0] == 0 and np.all(np.diff(starts) > 128)
        assert np.isfinite(eeg).all() and np.isfinite(kin).all()
        ends = np.r_[starts[1:], len(eeg)]
        eeg -= em
        eeg /= es
        kin -= km
        kin /= ks
        lifts = loadmat(data / f"{label}_AllLifts.mat", squeeze_me=True, struct_as_record=False)["P"].AllLifts
        assert set(lifts[:, 1].astype(int)) == set(range(1, 10))
        selected_runs = [8, 9] if participant == 1 else list(range(1, 10))
        # Replay P1 with the original concatenated test-batch boundaries.
        # GPU kernels can vary with batch shape; per-run calls shift those boundaries.
        control_predictions = None
        if participant == 1:
            positions = np.concatenate([np.arange(starts[r - 1], ends[r - 1] - 128, 64, dtype=np.int64) for r in selected_runs])
            control_x = np.stack([eeg[p:p + 128] for p in positions])
            control_predictions = {}
            for name, model in models.items():
                inp = control_x if name == "cnn_bilstm" else control_x[..., None]
                with tf.device("/GPU:0"):
                    control_predictions[name] = model.predict(inp, batch_size=64, verbose=0)
            del control_x, inp
        control_offset = 0
        ys, ids, indices = [], [], []
        predictions = {n: [] for n in NAMES}
        run_metrics, run_meta = {n: {} for n in NAMES}, {}
        for run in selected_runs:
            lo, hi = int(starts[run - 1]), int(ends[run - 1])
            pos = np.arange(lo, hi - 128, 64, dtype=np.int64)
            assert len(pos) and np.all(pos + 128 < hi)
            X = np.stack([eeg[p:p + 128] for p in pos])
            y = kin[pos + 128]
            pred = {"training_mean": np.broadcast_to(mean, y.shape).copy(),
                    "ridge": X.reshape(len(X), -1) @ coef.T + intercept}
            for name, model in models.items():
                inp = X if name == "cnn_bilstm" else X[..., None]
                if control_predictions is not None:
                    pred[name] = control_predictions[name][control_offset:control_offset + len(y)]
                else:
                    with tf.device("/GPU:0"):
                        pred[name] = model.predict(inp, batch_size=64, verbose=0)
            control_offset += len(y)
            for name in NAMES:
                run_metrics[name][str(run)] = scores(y, pred[name])
                predictions[name].append(pred[name])
            rows = lifts[lifts[:, 1].astype(int) == run]
            run_meta[str(run)] = {"samples": hi - lo, "n_windows": len(pos),
                                  "friction_levels": np.unique(rows[:, 4]).astype(int).tolist(),
                                  "lift_trials": len(rows)}
            ys.append(y); ids.append(np.full(len(y), run, dtype=np.int8)); indices.append(pos + 128)
            print("INFERENCE", label, "run", run, "windows", len(y), flush=True)
            del X, pred
        y, run_ids, target_indices = np.concatenate(ys), np.concatenate(ids), np.concatenate(indices)
        primary = np.isin(run_ids, [8, 9])
        entry = {"provenance": prov, "runs": run_meta, "models": {},
                 "data_file_sizes": {eeg_path.name: eeg_path.stat().st_size, kin_path.name: kin_path.stat().st_size},
                 "primary_friction_levels": sorted({v for r in ("8", "9") for v in run_meta[r]["friction_levels"]})}
        for name in NAMES:
            pred = np.concatenate(predictions[name])
            entry["models"][name] = {"primary_runs_8_9": scores(y[primary], pred[primary]),
                                      "all_evaluated_runs": scores(y, pred), "by_run": run_metrics[name]}
            path = out / f"{label}_{name}_predictions.npz"
            np.savez_compressed(path, y_true=y, y_pred=pred, run_id=run_ids, target_sample_index=target_indices)
            with np.load(path) as saved:
                check = scores(saved["y_true"][primary], saved["y_pred"][primary])
            for metric in ("mse", "mae", "r2"):
                np.testing.assert_allclose(check[metric], entry["models"][name]["primary_runs_8_9"][metric], rtol=1e-7, atol=1e-8)
            if participant == 1:
                with np.load(source / f"{name}_predictions.npz") as control:
                    np.testing.assert_array_equal(target_indices, control["target_sample_index"])
                    np.testing.assert_allclose(y, control["y_true"], rtol=1e-6, atol=1e-7)
                    np.testing.assert_allclose(pred, control["y_pred"], rtol=1e-3, atol=1e-2)
                    mean_difference = float(np.mean(np.abs(pred - control["y_pred"])))
                    assert mean_difference < 1e-3, (name, mean_difference)
                    entry["models"][name]["P1_control_max_absolute_prediction_difference"] = float(np.max(np.abs(pred - control["y_pred"])))
                    entry["models"][name]["P1_control_mean_absolute_prediction_difference"] = mean_difference
                for metric in ("mse", "mae", "r2"):
                    np.testing.assert_allclose(check[metric], original["models"][name]["test"][metric], rtol=1e-3, atol=1e-4)
            entry["models"][name]["saved_metric_check"] = "passed"
        for name, model in models.items():
            for before, after in zip(initial_weights[name], model.get_weights()):
                np.testing.assert_array_equal(before, after)
        result["participants"][label] = entry
        checkpoint()
        print("PARTICIPANT_COMPLETE", label, {n: entry["models"][n]["primary_runs_8_9"]["r2"] for n in NAMES}, flush=True)
        del eeg, kin, predictions, ys
    assert hashes == {f.name: digest(f) for f in artifacts}
    result["source_artifacts_unchanged"] = True
    result["all_neural_weights_unchanged"] = True
    result["P1_control_check"] = "passed"
    result["macro_summary"] = {}
    for name in NAMES:
        result["macro_summary"][name] = {}
        for scope in ("primary_runs_8_9", "all_evaluated_runs"):
            values = np.array([result["participants"][f"P{p}"]["models"][name][scope]["r2"] for p in range(2, 13)])
            result["macro_summary"][name][scope] = {"mean_r2": values.mean(axis=0).tolist(),
                "median_r2": np.median(values, axis=0).tolist(), "min_r2": values.min(axis=0).tolist(),
                "max_r2": values.max(axis=0).tolist(), "positive_participants_per_axis": (values > 0).sum(axis=0).tolist()}
    with (out / "participant_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["participant", "model", "scope", "windows", "r2_x", "r2_y", "r2_z", "mse_x", "mse_y", "mse_z", "mae_x", "mae_y", "mae_z"])
        for label, entry in result["participants"].items():
            for name in NAMES:
                for scope in ("primary_runs_8_9", "all_evaluated_runs"):
                    m = entry["models"][name][scope]
                    writer.writerow([label, name, scope, m["n_windows"], *m["r2"], *m["mse"], *m["mae"]])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax, name in zip(axes.flat, NAMES):
        values = np.array([result["participants"][f"P{p}"]["models"][name]["primary_runs_8_9"]["r2"] for p in range(1, 13)])
        for axis, axis_name in enumerate(("X", "Y", "Z")):
            ax.plot(range(1, 13), values[:, axis], marker="o", label=axis_name)
        ax.axhline(0, color="black", linewidth=.7)
        ax.axvline(1.5, color="gray", linestyle="--", linewidth=.7)
        ax.set(title=name, xlabel="Participant (P1 = source control)", ylabel="R² on runs 8–9", xticks=range(1, 13))
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle("Frozen P1 weights + P1 scalers: zero-adaptation transfer")
    fig.savefig(out / "cross_participant_r2.png", dpi=160); plt.close(fig)
    lines = ["# P1-trained, zero-adaptation cross-participant evaluation", "",
        "All six saved models and P1 training scalers are frozen. No target fitting, validation selection, batch-normalization updates or fine-tuning.",
        "Raw P1–P12 recordings verify wrist X/Y/Z provenance. Inference uses the same reduced data representation as the source experiment.",
        "Primary: runs 8–9 of P2–P12, with P1 runs 8–9 as a replay control. Secondary: all nine runs of each target participant.", "",
        "## Participant-macro R² on runs 8–9", "", "Each participant receives equal weight. P1 is excluded from the macro summary.", "",
        "| Model | Mean X | Mean Y | Mean Z | Median X | Median Y | Median Z |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        m = result["macro_summary"][name]["primary_runs_8_9"]
        lines.append("| " + name + " | " + " | ".join(f"{x:.4f}" for x in m["mean_r2"] + m["median_r2"]) + " |")
    lines += ["", "## Participant results on runs 8–9", "", "| Participant | Model | Windows | R² X | R² Y | R² Z |", "|---|---|---:|---:|---:|---:|"]
    for label, entry in result["participants"].items():
        for name in NAMES:
            m = entry["models"][name]["primary_runs_8_9"]
            lines.append(f"| {label} | {name} | {m['n_windows']} | " + " | ".join(f"{x:.4f}" for x in m["r2"]) + " |")
    lines += ["", "![Cross-participant R²](cross_participant_r2.png)", "", "## Verification", "",
        "P1 replay agrees with saved targets, predictions and scores. Metrics were recomputed from all saved prediction files. Neural weights and source artifacts remained unchanged.",
        "Complete per-run scores, friction metadata, provenance evidence and all-run summaries are in results.json; participant_metrics.csv provides a compact numeric table.",
        "Raw prediction arrays and model checkpoints remain local and ignored by Git.", "", "## Interpretation limits", ""]
    lines += ["- " + limit for limit in result["limits"]]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    result["evaluation_seconds"] = time.perf_counter() - started
    result["status"] = "complete"
    checkpoint()
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
