"""T1 validation-only comparison of participant-specific decoder gap preferences."""
import argparse
import ast
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for name, value in {
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    "OPENBLAS_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "4",
    "MPLBACKEND": "Agg",
}.items():
    os.environ.setdefault(name, value)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "outputs/timing-alignment-population-gpu-20261028T133000Z"
SIX_MODELS = ROOT / "experiments/02_model_comparison/compare_six_models.py"
NOTEBOOK = ROOT / "experiments/01_continuous_kinematic_decoding/corrected_baseline_v2.ipynb"
PROTOCOL = ROOT / "documentation/methods/T1_DECODER_GAP_PREFERENCES.md"
PEOPLE = tuple(f"P{i}" for i in range(1, 13))
GAPS = (2, 50, 100, 200, 500, 1000)
SEED, EPOCHS = 42, 15
NEW_MODELS = {"cnn_bilstm": "CNN–BiLSTM", "eegnet_regression": "EEGNet regression"}
LABELS = {"ridge": "Ridge", "cnn_elu": "CNN + ELU", **NEW_MODELS}
PARAMETERS = {"cnn_bilstm": 61347, "eegnet_regression": 1891}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_array(path, **values):
    import numpy as np

    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **values)
    tmp.replace(path)


def build_model(tf, model_name):
    """Reuse just the named factory from the historical notebook or runner."""
    if model_name == "cnn_bilstm":
        cells = read(NOTEBOOK)["cells"]
        source = next("".join(c["source"]) for c in cells
                      if c["cell_type"] == "code" and "def make_cnn_bilstm(" in "".join(c["source"]))
        origin = NOTEBOOK
    else:
        origin = SIX_MODELS
        source = origin.read_text(encoding="utf-8")
    factory_name = f"make_{model_name}"
    factory = next(node for node in ast.parse(source).body
                   if isinstance(node, ast.FunctionDef) and node.name == factory_name)
    scope = {name: getattr(tf.keras.layers, name) for name in
             ("Input", "Conv1D", "BatchNormalization", "Dropout", "Bidirectional", "LSTM", "Dense", "Activation")}
    scope.update(Sequential=tf.keras.Sequential, WINDOW=128, N_CHANNELS=32,
                 max_norm=tf.keras.constraints.max_norm)
    exec(compile(ast.Module(body=[factory], type_ignores=[]), str(origin), "exec"), scope)
    model = scope[factory_name]()
    assert model.count_params() == PARAMETERS[model_name]
    return model


def fit(out, person, gap):
    import numpy as np
    import run_population_timing as t1

    tf = t1.tf
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("T1 decoder fits require a TensorFlow GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    with tf.device("/GPU:0"):
        probe = tf.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
    assert "GPU:0" in probe.device
    probe.numpy()

    original = read(REFERENCE / "results.json")["participant_results"][person]
    assert all(digest(t1.REDUCED_DATA_DIR / filename) == sha
               for filename, sha in original["input_sha256"].items())
    eeg, target, bounds, splits, scale = t1.dataset(person)
    assert scale == original["scaling"]
    assert {str(k): list(v) for k, v in bounds.items()} == original["series_bounds"]
    with np.load(REFERENCE / person / "split_indices.npz") as frozen:
        for name in ("train", "validation"):
            indices, series = splits[name]
            np.testing.assert_array_equal(indices, frozen[f"{name}_targets"])
            np.testing.assert_array_equal(series, frozen[f"{name}_series_ids"])
    train = t1.make_windows(eeg, *splits["train"], bounds, gap // 2)
    val = t1.make_windows(eeg, *splits["validation"], bounds, gap // 2)
    y_train = target[splits["train"][0]]
    y_val = target[splits["validation"][0]]
    folder = out / person
    folder.mkdir(exist_ok=True)
    for model_name in NEW_MODELS:
        prefix = f"{model_name}_{gap}ms"
        path = folder / f"{prefix}.json"
        if path.exists():
            continue
        print("FIT_START", person, model_name, gap, stamp(), flush=True)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        x_train, x_val = (train, val) if model_name == "cnn_bilstm" else (train[..., None], val[..., None])
        with tf.device("/GPU:0"):
            model = build_model(tf, model_name)
            initialization = hashlib.sha256(b"".join(w.tobytes() for w in model.get_weights())).hexdigest()
            if gap != GAPS[0]:
                first = read(folder / f"{model_name}_{GAPS[0]}ms.json")
                assert initialization == first["initialization_sha256"]
            start = time.monotonic()
            history = model.fit(x_train, y_train, batch_size=64, epochs=EPOCHS, verbose=0)
            pred = model.predict(x_val, batch_size=64, verbose=0)
            weights = folder / f"{prefix}.keras"
            tmp_weights = folder / f"{prefix}.tmp.keras"
            model.save(tmp_weights)
        tmp_weights.replace(weights)
        assert y_val.shape == pred.shape and np.isfinite(pred).all()
        prediction_path = folder / f"{prefix}_validation.npz"
        save_array(prediction_path, y_true=y_val, y_pred=pred,
                   target_sample_index=splits["validation"][0], series_id=splits["validation"][1])
        mse = float(np.mean(np.square(y_val.astype(float) - pred.astype(float))))
        write(path, {
            "participant": person, "model": model_name, "gap_ms": gap, "seed": SEED,
            "epochs": EPOCHS, "initialization_sha256": initialization,
            "parameters": model.count_params(), "validation_mse": mse,
            "weights": weights.name, "weights_sha256": digest(weights),
            "predictions": prediction_path.name, "predictions_sha256": digest(prediction_path),
            "training_seconds": time.monotonic() - start,
            "loss_by_epoch": [float(v) for v in history.history["loss"]],
            "device": probe.device, "tensorflow": tf.__version__,
        })
        print("FIT_DONE", person, model_name, gap, mse, stamp(), flush=True)


def fit_path(out, person, model, gap):
    return out / person / f"{model}_{gap}ms.json"


def report(out, reference):
    import matplotlib.pyplot as plt
    import numpy as np

    counts = {model: dict.fromkeys(GAPS, 0) for model in LABELS}
    seed_mean_counts = dict.fromkeys(GAPS, 0)
    choices, candidate_rows = [], []
    for person in PEOPLE:
        folder = out / person
        original = reference["participant_results"][person]
        assert {str(gap // 2) for gap in GAPS} == set(original["gaps"])
        seed_mean = {
            gap: np.mean([fit["validation"]["mse_mean"] for fit in original["gaps"][str(gap // 2)]["cnn_elu"].values()])
            for gap in GAPS
        }
        legacy_gap = min(GAPS, key=lambda gap: (seed_mean[gap], gap))
        assert legacy_gap == reference["selection"][person]["cnn_elu"]["gap_ms"]
        seed_mean_counts[legacy_gap] += 1
        with np.load(REFERENCE / person / "split_indices.npz") as z:
            targets, ids = z["validation_targets"], z["validation_series_ids"]
        assert np.all(ids == 7)
        reference_y = None
        for model in LABELS:
            scores = {}
            initializations = set()
            for gap in GAPS:
                prior = original["gaps"][str(gap // 2)]
                if model in NEW_MODELS:
                    e = read(fit_path(out, person, model, gap))
                    assert (e["participant"], e["model"], e["gap_ms"], e["seed"], e["epochs"]) == (
                        person, model, gap, SEED, EPOCHS,
                    )
                    assert digest(folder / e["weights"]) == e["weights_sha256"]
                    assert digest(folder / e["predictions"]) == e["predictions_sha256"]
                    with np.load(folder / e["predictions"]) as z:
                        np.testing.assert_array_equal(targets, z["target_sample_index"])
                        np.testing.assert_array_equal(ids, z["series_id"])
                        y, pred = z["y_true"], z["y_pred"]
                    assert y.shape == pred.shape == (len(targets), 3) and np.isfinite(pred).all()
                    if reference_y is None:
                        reference_y = y
                    np.testing.assert_array_equal(y, reference_y)
                    val = float(np.mean(np.square(y.astype(float) - pred.astype(float))))
                    np.testing.assert_allclose(val, e["validation_mse"], atol=1e-12)
                    initializations.add(e["initialization_sha256"])
                else:
                    previous_fit = prior[model] if model == "ridge" else prior[model][str(SEED)]
                    val = previous_fit["validation"]["mse_mean"]
                scores[gap] = val
                candidate_rows.append({"participant": person, "decoder": model, "gap_ms": gap, "validation_mse": val})
            if model in NEW_MODELS:
                assert len(initializations) == 1
            first, second = sorted(GAPS, key=lambda gap: (scores[gap], gap))[:2]
            if model == "ridge":
                assert first == reference["selection"][person]["ridge"]["gap_ms"]
            counts[model][first] += 1
            choices.append({"participant": person, "decoder": model, "selected_gap_ms": first,
                            "validation_mse": scores[first], "runner_up_gap_ms": second,
                            "runner_up_mse": scores[second], "margin_mse": scores[second] - scores[first]})
    assert len(choices) == 48 and len(candidate_rows) == 288
    assert all(sum(row.values()) == len(PEOPLE) for row in counts.values())
    for filename, rows in (("validation_scores.csv", candidate_rows), ("selected_gaps.csv", choices)):
        with (out / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), sharey=True, layout="constrained")
    colors = ("#3b6b83", "#ad5d26", "#795693", "#287c70")
    for (model, label), ax, color in zip(LABELS.items(), axes.flat, colors):
        positions = np.arange(len(GAPS))
        heights = list(counts[model].values())
        bars = ax.bar(positions, heights, color=color, width=.63)
        ax.bar_label(bars, padding=3, fontsize=10)
        ax.set(xticks=positions, xticklabels=GAPS, ylim=(0, 13), yticks=range(0, 13, 2),
               title=label + (" · seed 42" if model != "ridge" else ""), xlabel="End gap (ms)")
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Participants selecting gap")
    fig.suptitle("T1 | Validation-selected EEG end gap for each decoder", fontsize=15, weight="bold")
    fig.savefig(out / "selected_gap_histograms.png", dpi=170)
    plt.close(fig)

    def modes(row):
        return [gap for gap, count in row.items() if count == max(row.values())]

    lines = ["# T1 — Validation gap preferences across decoders", "",
             "P1–P12, same 256 ms EEG inputs, recording splits and eligible targets as EXP-042. "
             "Every cell counts participant-specific choices on validation series 7. "
             "Neural model comparisons use seed 42.", "",
             "| Decoder | 2 ms | 50 ms | 100 ms | 200 ms | 500 ms | 1,000 ms | Most frequent gap |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    combined = {gap: sum(row[gap] for row in counts.values()) for gap in GAPS}
    for label, row in [(LABELS[m], c) for m, c in counts.items()] + [("All four decoders", combined)]:
        lines.append("| " + " | ".join([label, *[str(row[g]) for g in GAPS],
                                       ", ".join(f"{g} ms" for g in modes(row))]) + " |")
    lines += ["", "![Gap histograms](selected_gap_histograms.png)", "",
              "CNN + ELU’s earlier three-seed counts, in the same gap order, were "
              + ", ".join(str(seed_mean_counts[g]) for g in GAPS) + ".", "",
              "The combined row contains 48 participant–decoder choices. Repeated choices from the "
              "same person are dependent. The single neural seed and single validation series limit "
              "the precision of the histogram. The choices describe predictive alignment on validation.", "",
              "Verification checked all 144 new validation prediction arrays and model file hashes, "
              "alongside the shared T1 validation indices, saved errors and within-participant initialization hashes.", "",
              "[Protocol](declared_protocol.md) · [Gap choices and margins](selected_gaps.csv) · "
              "[Validation scores](validation_scores.csv) · [Verification](verification.json)", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    write(out / "gap_counts.json", {"per_decoder": counts, "combined": combined,
                                   "cnn_elu_three_seed_reference": seed_mean_counts})
    write(out / "verification.json", {"status": "passed", "participants": len(PEOPLE),
                                      "new_prediction_and_weight_files_checked": 144,
                                      "choices": len(choices), "validation_scores": len(candidate_rows)})


def supervise(out):
    sources = {
        "runner_source.py": Path(__file__), "declared_protocol.md": PROTOCOL,
        "t1_population_runner.py": Path(__file__).with_name("run_population_timing.py"),
        "t1_pilot_runner.py": Path(__file__).with_name("run_timing_alignment.py"),
        "timing_alignment_source.py": ROOT / "src/eeg_kinematics/timing_alignment.py",
        "six_model_source.py": SIX_MODELS, "corrected_baseline_v2.ipynb": NOTEBOOK,
    }
    for filename in ("results.json", "verification.json", "population_verification.json"):
        assert (REFERENCE / filename).is_file()
    assert all(read(REFERENCE / filename)["status"] == "passed"
               for filename in ("verification.json", "population_verification.json"))
    reference = read(REFERENCE / "results.json")
    hashes = {name: digest(p) for name, p in sources.items()}
    for key, filename in (("runner", "t1_population_runner.py"), ("pilot_runner", "t1_pilot_runner.py"),
                          ("timing_helper", "timing_alignment_source.py")):
        assert hashes[filename] == reference["source_sha256"][key]
    manifest = {"sources": hashes, "reference_results_sha256": digest(REFERENCE / "results.json")}
    manifest_path = out / "source_manifest.json"
    if manifest_path.exists():
        assert read(manifest_path) == manifest
    else:
        for name, p in sources.items():
            (out / name).write_bytes(p.read_bytes())
        write(manifest_path, manifest)
    status = out / "pipeline_status.json"

    def state(phase, **details):
        completed = sum(fit_path(out, p, m, g).is_file() for p in PEOPLE for m in NEW_MODELS for g in GAPS)
        write(status, {"stage": phase, "updated_at": stamp(), "supervisor_pid": os.getpid(),
                       "completed_fits": completed, "total_fits": 144, **details})

    try:
        for person in PEOPLE:
            for gap in GAPS:
                if all(fit_path(out, person, model, gap).is_file() for model in NEW_MODELS):
                    continue
                state("fitting", participant=person, gap_ms=gap)
                subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--output-dir",
                                str(out), "--worker", person, str(gap)], cwd=ROOT, check=True)
        state("verifying")
        report(out, reference)
        state("verified", report="RESULTS.md", histograms="selected_gap_histograms.png")
        print("T1_DECODER_GAP_PREFERENCES_VERIFIED", out, flush=True)
    except BaseException as error:
        state("failed", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.parent != (ROOT / "outputs").resolve() or not out.name.startswith("timing-decoder-preferences-"):
        parser.error("Choose a timing-decoder-preferences-* directory under outputs/")
    if args.worker:
        person, gap = args.worker
        assert person in PEOPLE and int(gap) in GAPS
        fit(out, person, int(gap))
        return
    out.mkdir(exist_ok=True)
    with (out / "training.log").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / "pipeline_status.json"
        if path.exists() and read(path)["stage"] == "verified":
            return
        supervise(out)


if __name__ == "__main__":
    main()

