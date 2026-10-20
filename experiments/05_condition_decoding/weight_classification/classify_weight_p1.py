"""C2: fixed-surface P1 weight classification, trial-grouped evaluation."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
import csv
import json
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
_repo = next(p for p in Path(__file__).resolve().parents if (p / "src/eeg_kinematics/paths.py").is_file())
sys.path.insert(0, str(_repo / "src"))
from eeg_kinematics.paths import REPO_ROOT, RAW_DATA_DIR, REDUCED_DATA_DIR

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import correlate
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

import importlib.util
_audit_spec = importlib.util.spec_from_file_location("condition_audit", REPO_ROOT / "experiments/00_dataset_audits/audit_trial_conditions.py")
_audit = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(_audit)
ROOT, WEIGHTS, metadata, sha = _audit.ROOT, _audit.WEIGHTS, _audit.metadata, _audit.sha

CLASSES = [165, 330, 660]
CODE_TO_CLASS = {1: 0, 2: 1, 4: 2}
SPLIT = {"train": [1, 4], "validation": [7], "test": [8, 9]}


def metrics(labels, prob):
    assert prob.shape == (len(labels), 3) and np.isfinite(prob).all()
    np.testing.assert_allclose(prob.sum(axis=1), 1, atol=1e-5)
    pred = prob.argmax(axis=1)
    cm = confusion_matrix(labels, pred, labels=[0, 1, 2])
    return {"n": len(labels), "accuracy": float(accuracy_score(labels, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "confusion_matrix": cm.tolist(), "class_recalls": (np.diag(cm) / cm.sum(axis=1)).tolist()}


def trial_average(prob, keys, trial_rows):
    averaged = np.stack([prob[keys == row["key"]].mean(axis=0) for row in trial_rows])
    labels = np.array([row["class_index"] for row in trial_rows])
    assert np.isfinite(averaged).all()
    return labels, averaged


def score_fast(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in [0, 1, 2]]))


def circular_reference(labels, prob, runs):
    """Conditional trial-label association reference, not model retraining."""
    pred = prob.argmax(axis=1)
    r8, r9 = np.flatnonzero(runs == 8), np.flatnonzero(runs == 9)
    assert len(r8) == len(r9) == 34
    observed = score_fast(labels, pred)
    null = []
    for a in range(34):
        for b in range(34):
            shifted = labels.copy()
            shifted[r8], shifted[r9] = np.roll(labels[r8], a), np.roll(labels[r9], b)
            null.append(score_fast(shifted, pred))
    null = np.array(null)
    p = float(np.mean(null >= observed - 1e-12))
    return {"type": "All within-run whole-trial circular label shifts with fixed predictions",
            "n_offsets": len(null), "observed_balanced_accuracy": observed,
            "mean_reference_balanced_accuracy": float(null.mean()),
            "reference_95th_percentile": float(np.quantile(null, .95)),
            "upper_tail_fraction": p, "bonferroni_two_classifiers": min(1., 2 * p)}, null


def make_cnn():
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, DepthwiseConv2D,
        Activation, Dropout, SeparableConv2D, AveragePooling2D, Flatten, Dense)
    model = Sequential([
        Input(shape=(128, 32, 1)),
        Conv2D(8, (64, 1), padding="same", use_bias=False), BatchNormalization(),
        DepthwiseConv2D((1, 32), use_bias=False, depth_multiplier=2),
        BatchNormalization(), Activation("elu"), Dropout(.25),
        SeparableConv2D(16, (16, 1), padding="same", use_bias=False),
        BatchNormalization(), Activation("elu"), AveragePooling2D((4, 1)), Dropout(.25),
        Flatten(), Dense(3, activation="softmax")])
    assert model.count_params() == 3235
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    args.audit_dir = args.audit_dir.resolve()
    assert args.audit_dir.parent == (ROOT / "outputs").resolve()
    audit = json.loads((args.audit_dir / "results.json").read_text())
    assert audit["status"] == "complete" and audit["C2_P1_plan"]["all_classes_present_in_each_selected_run"]
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("GPU required; no silent CPU fallback.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(42)
    with tf.device("/GPU:0"):
        probe = tf.linalg.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
    assert "GPU:0" in probe.device
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "outputs" / f"weight-classification-p1-gpu-{stamp}"
    out.mkdir(exist_ok=False)
    print("OUTPUT_DIRECTORY", out, flush=True)
    plan_path = ROOT / "documentation/methods/CONDITION_CLASSIFICATION.md"
    source_hashes = {"experiment": sha(Path(__file__)), "declared_protocol": sha(plan_path),
                     "condition_audit": sha(args.audit_dir / "results.json")}
    (out / "experiment_source.py").write_bytes(Path(__file__).read_bytes())
    (out / "declared_protocol.md").write_bytes(plan_path.read_bytes())
    report = {"status": "running", "experiment": "C2: P1 fixed-surface object-weight classification",
        "participant": "P1", "classes_grams": CLASSES, "surface": "sandpaper", "surface_code": 3,
        "runs": SPLIT, "trial_interval": "LEDOn inclusive to hand stop exclusive, full trial",
        "window_samples": 128, "stride_samples": 64, "sampling_rate": 500,
        "primary_unit": "trial; arithmetic mean of its window probabilities",
        "seed": 42, "neural_epochs": 15, "neural_batch_size": 64,
        "sample_weighting": "Equal total contribution per class; equal contribution per trial within each class",
        "audit_directory": args.audit_dir.relative_to(ROOT).as_posix(),
        "source_sha256": source_hashes, "models": {}, "clock_checks": {},
        "gpu": tf.config.experimental.get_device_details(gpus[0]), "gpu_probe_device": probe.device,
        "versions": {"tensorflow": tf.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "limits": ["Exploratory single participant and seed, with two training and two test runs.",
                   "Full-trial inference does not imply pre-contact weight anticipation or causal neural encoding.",
                   "Condition-associated movement, EEG artifacts, session drift and repeated conditions may contribute.",
                   "The reduced EEG upstream preprocessing remains undocumented.",
                   "Circular-shift reference preserves trial labels/counts under wrapping but assumes shift-exchangeable alignment.",
                   "Reference permutes evaluation labels only; it is not a full training-label permutation test.",
                   "No C3 condition-specific kinematics models or C5 stage-specific models are fitted."]}

    def save():
        (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    a, cols, metadata_path = metadata(1)
    report["raw_metadata_sha256"] = sha(metadata_path)
    data = REDUCED_DATA_DIR
    eeg = np.asarray(loadmat(data / "eeg_P1.mat", variable_names=["eeg"])["eeg"].T, dtype=np.float32, order="C")
    assert eeg.shape[1] == 32 and np.isfinite(eeg).all()
    starts = loadmat(data / "session_start_P1.mat")["session_start"].ravel().astype(int) - 1
    ends = np.r_[starts[1:], len(eeg)]
    trial_rows = []
    for split, runs in SPLIT.items():
        for run in runs:
            h = loadmat(RAW_DATA_DIR / f"P1/HS_P1_S{run}.mat", squeeze_me=True, struct_as_record=False)["hs"]
            w = loadmat(RAW_DATA_DIR / f"P1/WS_P1_S{run}.mat", squeeze_me=True, struct_as_record=False)["ws"]
            rows = a[a[:, cols["Run"]] == run]
            assert len(rows) == len(w.win) == 34 and h.eeg.samplingrate == 500
            assert len(h.eeg.sig) == ends[run - 1] - starts[run - 1]
            offsets = []
            for j in [0, 17, 33]:
                q = w.win[j]; v = q.eeg[:128, 0].astype(float); raw = h.eeg.sig[:, 0].astype(float)
                cost = np.convolve(raw * raw, np.ones(len(v)), mode="valid") - 2 * correlate(raw, v, mode="valid", method="fft") + np.sum(v * v)
                k = int(np.argmin(cost))
                np.testing.assert_array_equal(h.eeg.sig[k:k + 128], q.eeg[:128])
                offsets.append(float(q.trial_start_time - k / 500))
            assert np.ptp(offsets) < 1 / 500
            offset = float(np.median(offsets))
            for q, row in zip(w.win, rows):
                k = round((q.trial_start_time - offset) * 500)
                np.testing.assert_array_equal(h.eeg.sig[k:k + 100], q.eeg[:100])
                np.testing.assert_array_equal(h.kin.sig[k:k + 100, [21, 25, 29]], q.kin[:100, [21, 25, 29]])
                assert int(row[cols["BlockType"]]) == 1 and int(row[cols["CurS"]]) == 3
                assert str(q.surf_id) == "sandpaper" and int(q.weight) == int(row[cols["CurW"]])
                assert abs(q.trial_start_time - row[cols["StartTime"]]) < 1e-9
                lo, hi = [round((row[cols["StartTime"]] + row[cols[event]] - offset) * 500) for event in ["LEDOn", "tHandStop"]]
                assert 0 <= lo < hi <= len(h.eeg.sig)
                trial = int(row[cols["Lift"]]); weight = int(row[cols["CurW"]])
                trial_rows.append({"key": f"r{run}_t{trial}", "split": split, "run": run, "trial": trial,
                    "start_index": int(starts[run - 1] + lo), "end_index": int(starts[run - 1] + hi),
                    "class_index": CODE_TO_CLASS[weight], "weight_grams": WEIGHTS[weight],
                    "previous_weight_code": int(row[cols["PrevW"]]) if np.isfinite(row[cols["PrevW"]]) else None,
                    "duration_seconds": (hi - lo) / 500})
            report["clock_checks"][str(run)] = {"offset_seconds": offset, "all_trial_EEG_and_wrist_checks": 34}
            print("CLOCK_VERIFIED", run, offset, flush=True)
    scaler = StandardScaler()
    for row in trial_rows:
        if row["split"] == "train":
            scaler.partial_fit(eeg[row["start_index"]:row["end_index"]])
    expected_samples = sum(r["end_index"] - r["start_index"] for r in trial_rows if r["split"] == "train")
    assert scaler.n_samples_seen_ == expected_samples
    report["scaling"] = {"fit_runs": [1, 4], "fit_interval": "Training trials LEDOn to hand stop only",
                         "n_samples": expected_samples, "eeg_mean": scaler.mean_.tolist(), "eeg_scale": scaler.scale_.tolist()}
    partitions = {}
    for split in SPLIT:
        rr = [r for r in trial_rows if r["split"] == split]
        xs, labels, keys, positions = [], [], [], []
        for row in rr:
            pos = np.arange(row["start_index"], row["end_index"] - 128 + 1, 64, dtype=np.int64)
            assert len(pos) and np.all(pos + 128 <= row["end_index"])
            row["n_windows"] = len(pos)
            xs.extend(eeg[p:p + 128] for p in pos)
            labels.extend([row["class_index"]] * len(pos)); keys.extend([row["key"]] * len(pos)); positions.extend(pos)
        X = np.stack(xs)
        X -= scaler.mean_; X /= scaler.scale_
        partitions[split] = {"X": X, "y": np.array(labels), "keys": np.array(keys), "window_starts": np.array(positions), "trials": rr}
        assert set(labels) == {0, 1, 2}
        print("SPLIT", split, len(rr), "trials", len(X), "windows", flush=True)
    del eeg
    all_keys = [set(v["keys"].tolist()) for v in partitions.values()]
    assert all(not (all_keys[i] & all_keys[j]) for i in range(3) for j in range(i))
    report["split_counts"] = {s: {"trials": len(v["trials"]), "windows": len(v["X"]),
        "trials_per_class": dict(Counter(str(r["weight_grams"]) for r in v["trials"])),
        "duration_mean_by_class": {str(c): float(np.mean([r["duration_seconds"] for r in v["trials"] if r["weight_grams"] == c])) for c in CLASSES}} for s, v in partitions.items()}
    train, val, test = [partitions[s] for s in ["train", "validation", "test"]]
    counts = Counter(r["class_index"] for r in train["trials"])
    weights = np.concatenate([np.full(r["n_windows"], len(train["X"]) / (3 * counts[r["class_index"]] * r["n_windows"])) for r in train["trials"]])
    assert abs(weights.mean() - 1) < 1e-8
    feats = {s: np.concatenate([v["X"].mean(axis=1), v["X"].std(axis=1)], axis=1) for s, v in partitions.items()}
    feature_scaler = StandardScaler().fit(feats["train"])
    feats = {s: feature_scaler.transform(x) for s, x in feats.items()}
    candidates, best, best_score = [], None, -np.inf
    for C in [.01, .1, 1., 10.]:
        model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(feats["train"], train["y"], sample_weight=weights)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError(f"Logistic C={C} did not converge; do not accept its score.")
        yv, pv = trial_average(model.predict_proba(feats["validation"]), val["keys"], val["trials"])
        score = metrics(yv, pv)
        candidates.append({"C": C, "validation_trials": score, "iterations": model.n_iter_.tolist()})
        if score["balanced_accuracy"] > best_score:
            best, best_score = model, score["balanced_accuracy"]
        print("LOGISTIC", C, score["balanced_accuracy"], flush=True)
    report["models"]["logistic"] = {"device": "CPU", "features": "Per-channel window mean and standard deviation (64 features)",
        "selected_C": float(best.C), "selection": "Validation trial balanced accuracy; ties retain lower C; no refit on validation", "candidates": candidates}
    np.savez_compressed(out / "logistic_weights.npz", coef=best.coef_, intercept=best.intercept_,
                        feature_mean=feature_scaler.mean_, feature_scale=feature_scaler.scale_)
    save()
    model = make_cnn()
    started = time.perf_counter()
    with tf.device("/GPU:0"):
        history = model.fit(train["X"][..., None], train["y"], sample_weight=weights,
            validation_data=(val["X"][..., None], val["y"]), epochs=15, batch_size=64, verbose=2)
        model.save(out / "cnn_elu_classifier.keras")
    report["models"]["cnn_elu"] = {"device": "GPU", "parameters": model.count_params(),
        "epochs": 15, "fit_seconds": time.perf_counter() - started, "checkpoint": "Fixed final epoch",
        "history": {k: [float(x) for x in v] for k, v in history.history.items()}}
    model = tf.keras.models.load_model(out / "cnn_elu_classifier.keras", compile=False)
    model.trainable = False
    majority = int(max(counts, key=counts.get))
    report["models"]["training_majority"] = {"device": "CPU", "class_grams": CLASSES[majority], "fit": "Training trials only"}
    for split in ["validation", "test"]:
        v = partitions[split]
        probs = {"logistic": best.predict_proba(feats[split]), "training_majority": np.eye(3)[np.full(len(v["y"]), majority)]}
        with tf.device("/GPU:0"):
            probs["cnn_elu"] = model.predict(v["X"][..., None], batch_size=64, verbose=0)
        for name, prob in probs.items():
            ytrial, ptrial = trial_average(prob, v["keys"], v["trials"])
            runs = np.array([r["run"] for r in v["trials"]])
            entry = {"trial": metrics(ytrial, ptrial), "window_secondary": metrics(v["y"], prob),
                     "by_run": {str(r): metrics(ytrial[runs == r], ptrial[runs == r]) for r in np.unique(runs)}}
            if split == "test" and name != "training_majority":
                entry["circular_shift_reference"], null = circular_reference(ytrial, ptrial, runs)
                np.savez_compressed(out / f"{name}_circular_reference.npz", balanced_accuracy=null)
            report["models"][name][split] = entry
            np.savez_compressed(out / f"{name}_{split}_predictions.npz", window_y=v["y"], window_prob=prob,
                trial_key=v["keys"], window_start=v["window_starts"], trial_y=ytrial, trial_prob=ptrial, trial_run=runs,
                trial_number=np.array([r["trial"] for r in v["trials"]]))
            print("EVALUATED", name, split, entry["trial"], flush=True)
    (out / "trial_manifest.json").write_text(json.dumps(trial_rows, indent=2) + "\n")
    with (out / "trial_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["split", "run", "trial", "model", "weight_grams", "predicted_grams", "p165", "p330", "p660"])
        for split in ["validation", "test"]:
            for name in report["models"]:
                with np.load(out / f"{name}_{split}_predictions.npz") as z:
                    for run, tr, y, prob in zip(z["trial_run"], z["trial_number"], z["trial_y"], z["trial_prob"]):
                        writer.writerow([split, run, tr, name, CLASSES[y], CLASSES[prob.argmax()], *prob])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, name in zip(axes, ["training_majority", "logistic", "cnn_elu"]):
        cm = np.array(report["models"][name]["test"]["trial"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for i in range(3):
            for j in range(3): ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(title=name, xticks=range(3), xticklabels=CLASSES, yticks=range(3), yticklabels=CLASSES,
               xlabel="Predicted weight (g)", ylabel="Observed weight (g)")
    fig.suptitle("P1 held-out runs 8–9: one prediction per trial")
    fig.savefig(out / "trial_confusion_matrices.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, name in zip(axes, ["logistic", "cnn_elu"]):
        with np.load(out / f"{name}_circular_reference.npz") as z: ax.hist(z["balanced_accuracy"], bins=20, alpha=.75)
        ax.axvline(report["models"][name]["test"]["trial"]["balanced_accuracy"], color="red", label="Observed")
        ax.set(title=name, xlabel="Trial balanced accuracy", ylabel="Circular label offsets"); ax.legend()
    fig.savefig(out / "circular_shift_reference.png", dpi=160); plt.close(fig)
    assert source_hashes == {"experiment": sha(Path(__file__)), "declared_protocol": sha(plan_path), "condition_audit": sha(args.audit_dir / "results.json")}
    report["source_files_unchanged"] = True
    report["status"] = "complete"
    save()
    lines = ["# C2: P1 EEG weight classification at fixed sandpaper", "",
        "Train runs 1/4 (68 trials), validate run 7 (34), test runs 8/9 (68). Full LEDOn-to-hand-stop intervals; 128-sample/stride-64 windows stay within trials.",
        "Training-only EEG scaling. Trial/class-balanced training weights. Mean window probabilities produce one trial prediction.", "",
        "| Model | Trial accuracy | Trial balanced accuracy | Macro F1 | Circular upper-tail fraction | Adjusted across two classifiers |", "|---|---:|---:|---:|---:|---:|"]
    for name, item in report["models"].items():
        m = item["test"]["trial"]; ref = item["test"].get("circular_shift_reference")
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {m['macro_f1']:.4f} | " +
                     (f"{ref['upper_tail_fraction']:.4f} | {ref['bonferroni_two_classifiers']:.4f}" if ref else "— | —") + " |")
    lines += ["", "Balanced-accuracy chance reference: 1/3. Window-level scores are secondary, not independent sample evidence.",
        f"Logistic C={best.C:g} uses validation trial balanced accuracy. CNN + ELU uses fixed final epoch 15, seed 42, on GPU.",
        "", "![Confusion matrices](trial_confusion_matrices.png)", "", "![Circular reference](circular_shift_reference.png)",
        "", "## Reference and interpretation", "",
        "All 1,156 within-run trial-label offsets are enumerated with predictions fixed. This measures evaluation-label association, not training under shuffled labels. Circular exchangeability is an assumption, not an established property of these recordings.",
        "Class counts, trial durations, per-run scores, exact clock checks and validation selection are in results.json. No classifier receives kinematics, trial time, run ID or previous-condition metadata as input.",
        "", "## Limits", ""]
    lines += ["- " + s for s in report["limits"]]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
