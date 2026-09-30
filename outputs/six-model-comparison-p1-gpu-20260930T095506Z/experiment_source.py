"""Fixed P1 v2 protocol: two basic baselines and four neural networks.

Run from the repository with the existing WSL TensorFlow GPU Python.
Notebook preparation is reused verbatim; the original notebooks are not edited.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import hashlib
import json
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import Activation
from tensorflow.keras.constraints import max_norm

PROJECT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--resume", type=Path, help="Resume an incomplete output directory; preserve completed fits.")
args = parser.parse_args()
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError("GPU required for this experiment; no silent CPU fallback.")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
with tf.device("/GPU:0"):
    probe = tf.linalg.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
assert "GPU:0" in probe.device, probe.device
print("GPU computation verified:", probe.device, flush=True)

notebook_path = PROJECT / "pipeline_v2.ipynb"
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
scope = {"PARTICIPANT_ID": 1}
preparation = []
model_cell = None
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    if source.startswith("PARTICIPANT_ID ="):
        continue
    if "def make_cnn():" in source:
        model_cell = source
        break
    preparation.append(source)
    exec(compile(source, str(notebook_path), "exec"), scope)
assert model_cell and scope["PARTICIPANT"] == "P1"
assert scope["TRAIN_RUNS"] == (1, 2, 3, 4, 5, 6)
assert scope["VAL_RUNS"] == (7,) and scope["TEST_RUNS"] == (8, 9)
exec(compile(model_cell, str(notebook_path), "exec"), scope)

X_train, y_train = scope["X_train"], scope["y_train"]
X_val, y_val = scope["X_val"], scope["y_val"]
X_test, y_test = scope["X_test"], scope["y_test"]
test_ids, test_targets = scope["test_ids"], scope["test_targets"]
assert X_train.shape == (18516, 128, 32)
assert X_val.shape == (1867, 128, 32) and X_test.shape == (3642, 128, 32)

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
if args.resume:
    run_dir = args.resume.resolve()
    assert run_dir.parent == (PROJECT / "outputs").resolve()
    prior_results = json.loads((run_dir / "results.json").read_text())
    assert prior_results["status"] == "running" and prior_results["subject"] == "P1"
    assert prior_results["notebook_sha256"] == hashlib.sha256(notebook_path.read_bytes()).hexdigest()
else:
    run_dir = PROJECT / "outputs" / f"six-model-comparison-p1-gpu-{stamp}"
    run_dir.mkdir(exist_ok=False)
(run_dir / "source_notebook.ipynb").write_bytes(notebook_path.read_bytes())
(run_dir / "experiment_source.py").write_bytes(Path(__file__).read_bytes())
print("OUTPUT_DIRECTORY", run_dir, flush=True)
try:
    revision = subprocess.check_output(["git", "-c", f"safe.directory={PROJECT}", "rev-parse", "HEAD"], text=True).strip()
    git_status = subprocess.check_output(["git", "-c", f"safe.directory={PROJECT}", "status", "--short"], text=True)
except subprocess.CalledProcessError:
    revision, git_status = None, "unavailable"

def scores(true, pred):
    assert true.shape == pred.shape and np.isfinite(pred).all()
    return {
        "mse": mean_squared_error(true, pred, multioutput="raw_values").tolist(),
        "mae": mean_absolute_error(true, pred, multioutput="raw_values").tolist(),
        "r2": r2_score(true, pred, multioutput="raw_values").tolist(),
    }

results = {
    "status": "running", "subject": "P1", "seed": 42,
    "protocol": "Six-model comparison using fixed v2 P1 run split",
    "runs": {"train": list(scope["TRAIN_RUNS"]), "validation": [7], "test": [8, 9]},
    "split_shapes": {"train": list(X_train.shape), "validation": list(X_val.shape), "test": list(X_test.shape)},
    "window_samples": 128, "stride_samples": 64, "target_lead_after_last_input_samples": 1,
    "friction_levels_by_split": scope["FRICTION_BY_SPLIT"],
    "target_axes": list(scope["AXES"]), "metric_units": "train-standardized target units",
    "scaling": {"fit_runs": list(scope["TRAIN_RUNS"]), "eeg_mean": scope["eeg_scaler"].mean_.tolist(),
                "eeg_scale": scope["eeg_scaler"].scale_.tolist(),
                "target_mean": scope["kin_scaler"].mean_.tolist(), "target_scale": scope["kin_scaler"].scale_.tolist()},
    "versions": {"python": scope["sys"].version.split()[0], "tensorflow": tf.__version__,
                 "keras": tf.keras.__version__, "numpy": np.__version__,
                 "scipy": scope["scipy"].__version__, "sklearn": scope["sklearn"].__version__},
    "gpu": tf.config.experimental.get_device_details(gpus[0]), "gpu_probe_device": probe.device,
    "git_revision": revision, "git_status_at_start": git_status,
    "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
    "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "eegnet_source": "https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py",
    "notes": ["Single seed; GPU arithmetic may vary.", "P1 test runs contain friction 3 only.",
              "Existing neural architectures are retrained, not reused from the previous result.",
              "No test-based hyperparameter or checkpoint selection.",
              "EEGNet regression preserves default sample kernels, ELU, pooling and constraints; only output is adapted.",
              "This is an EEGNet architecture reference, not a reproduction of its published classification experiments."],
    "models": {},
}
if args.resume:
    results["models"] = prior_results["models"]
    results["resume_note"] = "Resumed completed original-model fits; corrected EEGNet spatial max-norm axis before its training."

def checkpoint():
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

checkpoint()
# Fit both basic models without touching test scores.
started = time.perf_counter()
train_mean = y_train.mean(axis=0, dtype=np.float64)
np.savez_compressed(run_dir / "training_mean_weights.npz", mean=train_mean)
results["models"]["training_mean"] = {
    "device": "cpu", "parameters": 3, "training_seconds": time.perf_counter() - started,
    "validation": scores(y_val, np.broadcast_to(train_mean, y_val.shape)),
}
flat_train = X_train.reshape(len(X_train), -1)
flat_val = X_val.reshape(len(X_val), -1)
started = time.perf_counter()
ridge_candidates = []
ridge_model = None
best_mse = float("inf")
for alpha in (1.0, 10.0, 100.0, 1000.0, 10000.0):
    candidate = Ridge(alpha=alpha, solver="lsqr", tol=1e-3, max_iter=200)
    candidate.fit(flat_train, y_train)
    validation_mse = float(mean_squared_error(y_val, candidate.predict(flat_val)))
    ridge_candidates.append({"alpha": alpha, "validation_mse": validation_mse,
                             "n_iter": candidate.n_iter_.tolist()})
    print("RIDGE", ridge_candidates[-1], flush=True)
    if validation_mse < best_mse:
        best_mse, ridge_model = validation_mse, candidate
assert ridge_model is not None
np.savez_compressed(run_dir / "ridge_weights.npz", coef=ridge_model.coef_, intercept=ridge_model.intercept_)
results["models"]["ridge"] = {
    "device": "cpu", "parameters": int(ridge_model.coef_.size + ridge_model.intercept_.size),
    "training_seconds": time.perf_counter() - started, "selected_alpha": ridge_model.alpha,
    "selection": "minimum validation-run MSE; no refit on validation",
    "solver": "lsqr", "tol": 1e-3, "max_iter": 200, "candidates": ridge_candidates,
    "validation": scores(y_val, ridge_model.predict(flat_val)),
}
checkpoint()

# ELU-only: exact current CNN with two inserted hidden activations.
def make_cnn_elu():
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, DepthwiseConv2D,
        Dropout, SeparableConv2D, AveragePooling2D, Flatten, Dense)
    model = Sequential([
        Input(shape=(128, 32, 1)),
        Conv2D(8, (64, 1), padding="same", use_bias=False), BatchNormalization(),
        DepthwiseConv2D((1, 32), use_bias=False, depth_multiplier=2),
        BatchNormalization(), Activation("elu"), Dropout(0.25),
        SeparableConv2D(16, (16, 1), use_bias=False, padding="same"),
        BatchNormalization(), Activation("elu"), AveragePooling2D((4, 1)), Dropout(0.25),
        Flatten(), Dense(3),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    assert model.count_params() == 3235
    return model

def make_eegnet_regression():
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, DepthwiseConv2D,
        Dropout, SeparableConv2D, AveragePooling2D, Flatten, Dense)
    # Time-first equivalent of authors' channel-first input geometry.
    model = Sequential([
        Input(shape=(128, 32, 1)),
        Conv2D(8, (64, 1), padding="same", use_bias=False), BatchNormalization(),
        DepthwiseConv2D((1, 32), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1., axis=1)),
        BatchNormalization(), Activation("elu"), AveragePooling2D((4, 1)), Dropout(0.5),
        SeparableConv2D(16, (16, 1), use_bias=False, padding="same"),
        BatchNormalization(), Activation("elu"), AveragePooling2D((8, 1)), Dropout(0.5),
        Flatten(), Dense(3, kernel_constraint=max_norm(0.25)),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    assert model.count_params() == 1891
    return model

definitions = [
    ("cnn", scope["make_cnn"], 15, True),
    ("cnn_bilstm", scope["make_cnn_bilstm"], 20, False),
    ("cnn_elu", make_cnn_elu, 15, True),
    ("eegnet_regression", make_eegnet_regression, 15, True),
]
for name, factory, epochs, extra_axis in definitions:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    with tf.device("/GPU:0"):
        model = factory()
        # Same seeded weight initialization for the affine and ELU-only CNNs.
        if name == "cnn":
            cnn_initial = [w.copy() for w in model.get_weights()]
        if name == "cnn_elu":
            assert all(np.array_equal(a, b) for a, b in zip(cnn_initial, model.get_weights()))
        if args.resume and name in results["models"] and (run_dir / f"{name}.keras").is_file():
            print("REUSING completed fit", name, flush=True)
            continue
        train_input = X_train[..., None] if extra_axis else X_train
        val_input = X_val[..., None] if extra_axis else X_val
        started = time.perf_counter()
        print("TRAINING", name, "epochs", epochs, "parameters", model.count_params(), flush=True)
        history = model.fit(train_input, y_train, epochs=epochs, batch_size=64,
                            validation_data=(val_input, y_val), verbose=2)
        training_seconds = time.perf_counter() - started
        val_pred = model.predict(val_input, batch_size=64, verbose=0)
        model.save(run_dir / f"{name}.keras")
    results["models"][name] = {
        "device": "gpu", "parameters": model.count_params(), "epochs": epochs, "batch_size": 64,
        "training_seconds": training_seconds, "seed": 42,
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "validation": scores(y_val, val_pred),
        "checkpoint": "fixed final epoch",
    }
    checkpoint()

# All fitting and selection is finished. Evaluate every model on the same test windows.
predictions = {
    "training_mean": np.broadcast_to(train_mean, y_test.shape).copy(),
    "ridge": ridge_model.predict(X_test.reshape(len(X_test), -1)),
}
for name, _, _, extra_axis in definitions:
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(run_dir / f"{name}.keras", compile=False)
    test_input = X_test[..., None] if extra_axis else X_test
    with tf.device("/GPU:0"):
        predictions[name] = model.predict(test_input, batch_size=64, verbose=0)
        repeat = model(test_input[:64], training=False).numpy()
    # GPU graph/direct-call kernels can use different floating-point reduction
    # orders. Compare within a small standardized-target tolerance, not bitwise.
    np.testing.assert_allclose(repeat, predictions[name][:64], rtol=1e-3, atol=5e-4)
    results["models"][name]["reload_prediction_check"] = {
        "status": "passed", "rtol": 1e-3, "atol": 5e-4,
        "max_absolute_difference": float(np.max(np.abs(repeat - predictions[name][:64]))),
    }

for name, pred in predictions.items():
    entry = results["models"][name]
    entry["test"] = scores(y_test, pred)
    entry["by_test_run"] = {str(run): {"n_windows": int((test_ids == run).sum()),
        **scores(y_test[test_ids == run], pred[test_ids == run])} for run in (8, 9)}
    np.savez_compressed(run_dir / f"{name}_predictions.npz", y_true=y_test, y_pred=pred,
                        run_id=test_ids, target_sample_index=test_targets)
    # Confirm public metrics are recoverable from saved arrays.
    with np.load(run_dir / f"{name}_predictions.npz") as saved:
        reread = scores(saved["y_true"], saved["y_pred"])
    for metric in ("mse", "mae", "r2"):
        np.testing.assert_allclose(reread[metric], entry["test"][metric], rtol=1e-6, atol=1e-7)
    entry["saved_metric_check"] = "passed"
    print("TEST", name, entry["test"], flush=True)

import matplotlib.pyplot as plt
names = list(predictions)
fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
for axis, ax in enumerate(axes):
    ax.barh(names, [results["models"][n]["test"]["r2"][axis] for n in names])
    ax.axvline(0, color="black", linewidth=.7)
    ax.set(title=scope["AXES"][axis], xlabel="Held-out R²")
fig.suptitle("P1 · runs 8–9 · seed 42 · six models")
fig.savefig(run_dir / "r2_comparison.png", dpi=160)
plt.close(fig)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
for ax, (name, _, _, _) in zip(axes.flat, definitions):
    history = results["models"][name]["history"]
    ax.plot(range(1, len(history["loss"]) + 1), history["loss"], label="Train (dropout active)")
    ax.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="Validation run 7")
    ax.set(title=name, xlabel="Epoch", ylabel="MSE (standardized targets)")
    ax.legend()
fig.savefig(run_dir / "learning_curves.png", dpi=160)
plt.close(fig)

lines = ["# Six-model comparison — P1", "",
    "Train runs 1–6; validation run 7; test runs 8–9. Same v2 train-only scaling and 128-sample/stride-64 windows for every model.",
    "Four neural models trained on the GPU; mean and ridge on CPU. Neural seed 42; fixed final epochs (CNN variants 15, CNN–BiLSTM 20).", "",
    "| Model | Parameters | Fit seconds | R² X | R² Y | R² Z |", "|---|---:|---:|---:|---:|---:|"]
for name in names:
    m = results["models"][name]
    lines.append(f"| {name} | {m['parameters']} | {m['training_seconds']:.2f} | " + " | ".join(f"{v:.4f}" for v in m["test"]["r2"]) + " |")
lines += ["", f"Ridge alpha {ridge_model.alpha:g} was selected from 1, 10, 100, 1000, 10000 using validation MSE only.",
    "ELU-only CNN adds two activations, keeping the original weights' initialization, pooling, dropout, capacity and schedule.",
    "EEGNet regression follows the authors' feature extractor with a linear three-axis head: ELU, 4× then 8× pooling, dropout 0.5, and max-norm constraints.",
    "Its 1,891 parameters differ from the 3,235-parameter CNN. It preserves sample-based kernels; sampling-rate retuning was not performed.", "",
    "## Held-out runs separately", "", "| Model | Run | R² X | R² Y | R² Z |", "|---|---:|---:|---:|---:|"]
for name in names:
    for run in (8, 9):
        lines.append(f"| {name} | {run} | " + " | ".join(f"{v:.4f}" for v in results["models"][name]["by_test_run"][str(run)]["r2"]) + " |")
lines += ["", "![R² comparison](r2_comparison.png)", "", "![Learning curves](learning_curves.png)", "",
    "Single participant and seed; test friction condition 3 only. This does not establish cross-participant generalization or statistical significance.",
    "Positive/negative R² is relative to the test mean; the training-mean baseline can therefore have negative R².",
    "MSE and MAE use standardized targets; complete values and validation selection are in results.json.",
    "Saved neural models were reloaded for test inference; saved prediction metrics were recomputed successfully.",
    "Original notebooks and historical results are preserved. Model weights, predictions and notebook snapshots remain local.", ""]
(run_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
results["status"] = "complete"
checkpoint()
print("COMPLETE", run_dir, flush=True)
