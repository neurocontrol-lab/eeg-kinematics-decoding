"""Score the added T1 decoders at their frozen validation choice and at 2 ms."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import run_decoder_gap_preferences as sweep

REFERENCE_SUMMARY = sweep.REFERENCE / "population_summary.json"
BASELINE_GAP = sweep.GAPS[0]


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def frozen_choices(run):
    checks = sweep.read(run / "verification.json")
    assert sweep.read(run / "pipeline_status.json")["stage"] == "verified"
    assert checks["status"] == "passed" and checks["new_prediction_and_weight_files_checked"] == 144
    rows = csv_rows(run / "validation_scores.csv")
    saved = {(r["participant"], r["decoder"]): int(r["selected_gap_ms"])
             for r in csv_rows(run / "selected_gaps.csv")}
    choices = {}
    for person in sweep.PEOPLE:
        choices[person] = {}
        for model in sweep.NEW_MODELS:
            scores = {int(r["gap_ms"]): float(r["validation_mse"]) for r in rows
                      if (r["participant"], r["decoder"]) == (person, model)}
            assert set(scores) == set(sweep.GAPS)
            gap = min(scores, key=lambda g: (scores[g], g))
            assert saved[person, model] == gap
            choices[person][model] = gap
    return {
        "choices": choices,
        "validation_scores_sha256": sweep.digest(run / "validation_scores.csv"),
        "selected_gaps_sha256": sweep.digest(run / "selected_gaps.csv"),
        "validation_verification_sha256": sweep.digest(run / "verification.json"),
        "original_population_summary_sha256": sweep.digest(REFERENCE_SUMMARY),
    }


def wait_for_validation(run):
    status_path = run / "pipeline_status.json"
    with (run / "training.log").open("rb") as validation_log:
        while True:
            stage = sweep.read(status_path)["stage"]
            if stage == "verified":
                return
            if stage == "failed":
                raise RuntimeError("Validation sweep failed. Inspect its training log before resuming.")
            try:
                fcntl.flock(validation_log, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(30)
            else:
                # Re-read after acquiring the lock in case the writer just finished.
                stage = sweep.read(status_path)["stage"]
                if stage == "verified":
                    return
                raise RuntimeError(f"Validation supervisor has stopped at {stage}.")


def score_participant(run, out, person):
    import numpy as np
    import run_population_timing as t1

    tf = t1.tf
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("TensorFlow GPU unavailable")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    with tf.device("/GPU:0"):
        probe = tf.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
    assert "GPU:0" in probe.device
    probe.numpy()

    frozen = sweep.read(out / "selection.json")["choices"][person]
    source = sweep.read(sweep.REFERENCE / "results.json")["participant_results"][person]
    assert all(sweep.digest(t1.REDUCED_DATA_DIR / name) == checksum
               for name, checksum in source["input_sha256"].items())
    eeg, target, bounds, splits, scaling = t1.dataset(person)
    assert scaling == source["scaling"]
    indices, ids = splits["test"]
    with np.load(sweep.REFERENCE / person / "split_indices.npz") as ref:
        np.testing.assert_array_equal(indices, ref["test_targets"])
        np.testing.assert_array_equal(ids, ref["test_series_ids"])
    assert set(ids) == {8, 9}
    y = target[indices]
    folder = out / person
    folder.mkdir(exist_ok=True)
    windows = {g: t1.make_windows(eeg, indices, ids, bounds, g // 2)
               for g in {BASELINE_GAP, *frozen.values()}}
    result = {}
    for model, selected in frozen.items():
        result[model] = {}
        for gap in sorted({BASELINE_GAP, selected}):
            fit = sweep.read(sweep.fit_path(run, person, model, gap))
            weights = run / person / fit["weights"]
            assert fit["weights_sha256"] == sweep.digest(weights)
            x = windows[gap] if model == "cnn_bilstm" else windows[gap][..., None]
            tf.keras.backend.clear_session()
            with tf.device("/GPU:0"):
                net = tf.keras.models.load_model(weights, compile=False)
                pred = net.predict(x, batch_size=64, verbose=0)
            del net
            assert pred.shape == y.shape and np.isfinite(pred).all()
            path = folder / f"{model}_{gap}ms_test.npz"
            sweep.save_array(path, y_true=y, y_pred=pred, target_sample_index=indices, series_id=ids)
            result[model][str(gap)] = {"file": path.name, "sha256": sweep.digest(path),
                                       "weights_sha256": fit["weights_sha256"]}
    sweep.write(folder / "test.json", result)
    print("TEST_SCORED", person, sweep.stamp(), flush=True)


def summarize(out, frozen):
    import numpy as np
    import run_population_timing as t1

    original = sweep.read(REFERENCE_SUMMARY)
    assert sweep.digest(REFERENCE_SUMMARY) == frozen["original_population_summary_sha256"]
    rows, population = original["participant_metrics"], original["population"]
    new_choices = frozen["choices"]
    samples = np.random.default_rng(20261028).integers(0, len(sweep.PEOPLE), size=(10000, len(sweep.PEOPLE)))
    checked = 0
    for person in sweep.PEOPLE:
        with np.load(sweep.REFERENCE / person / "ridge_gap1_predictions.npz") as ref:
            indices, ids, truth = ref["target_sample_index"], ref["series_id"], ref["y_true"]
        folder = out / person
        data = sweep.read(folder / "test.json")
        assert set(data) == set(sweep.NEW_MODELS)
        for model, gap in new_choices[person].items():
            paths = data[model]
            assert set(paths) == {str(BASELINE_GAP), str(gap)}
            scores = {}
            for key, metadata in paths.items():
                path = folder / metadata["file"]
                assert sweep.digest(path) == metadata["sha256"]
                with np.load(path) as z:
                    np.testing.assert_array_equal(z["target_sample_index"], indices)
                    np.testing.assert_array_equal(z["series_id"], ids)
                    np.testing.assert_array_equal(z["y_true"], truth)
                    scores[key] = t1.scored(truth, z["y_pred"], ids)
                checked += 1
            b, s = (scores[str(g)] for g in (BASELINE_GAP, gap))
            ref_mse, sel_mse = (v["aggregate"]["mse_mean"] for v in (b, s))
            delta = sel_mse - ref_mse
            reduction = -100 * delta / ref_mse
            row = {
                "participant": person, "model": model, "selected_gap_ms": gap,
                "test_targets": len(indices), "baseline_mse": ref_mse, "selected_mse": sel_mse,
                "selected_minus_baseline": delta, "percent_mse_reduction": reduction,
                "seed_mse_differences": [delta], "seed_percent_reductions": [reduction],
            }
            for tag, metric in (("baseline", b), ("selected", s)):
                for key in ("mse", "rmse", "mae", "r2"):
                    row[f"{tag}_axis_{key}"] = metric["aggregate"][key]
            row["by_series"] = {}
            for series in ("8", "9"):
                x, y = (v["by_series"][series]["mse_mean"] for v in (b, s))
                row["by_series"][series] = {
                    "baseline_mse": x, "selected_mse": y, "selected_minus_baseline": y-x,
                }
            rows.append(row)

    for model in sweep.NEW_MODELS:
        subset = [r for r in rows if r["model"] == model]
        assert len(subset) == len(sweep.PEOPLE)
        deltas = np.array([r["selected_minus_baseline"] for r in subset])
        reductions = np.array([r["percent_mse_reduction"] for r in subset])
        population[model] = {
            "participants": len(subset),
            "mean_baseline_mse": float(np.mean([r["baseline_mse"] for r in subset])),
            "mean_selected_mse": float(np.mean([r["selected_mse"] for r in subset])),
            "mean_selected_minus_baseline": float(deltas.mean()),
            "paired_mse_difference_95_interval": np.quantile(deltas[samples].mean(axis=1), [.025, .975]).tolist(),
            "mean_participant_percent_reduction": float(reductions.mean()),
            "percent_reduction_95_interval": np.quantile(reductions[samples].mean(axis=1), [.025, .975]).tolist(),
            "median_participant_percent_reduction": float(np.median(reductions)),
            "participants_improved": int((deltas < 0).sum()),
            "participants_unchanged": int((deltas == 0).sum()),
            "participants_worse": int((deltas > 0).sum()),
            "selected_gap_counts": {str(g): sum(r["selected_gap_ms"] == g for r in subset)
                                    for g in sweep.GAPS},
        }
    original["seeds_by_decoder"] = {"ridge": [], "cnn_elu": [42, 43, 44],
                                    **dict.fromkeys(sweep.NEW_MODELS, [sweep.SEED])}
    assert len(rows) == 48
    sweep.write(out / "population_summary.json", original)
    sweep.write(out / "verification.json", {
        "status": "passed", "participants": len(sweep.PEOPLE), "new_predictions_checked": checked,
        "participant_decoder_comparisons": len(rows), "models_fitted": 0,
    })


def supervise(run, out, wait):
    def update(stage, **fields):
        sweep.write(out / "pipeline_status.json", {"stage": stage, "updated_at": sweep.stamp(),
                                                   "pid": os.getpid(), **fields})

    try:
        if wait:
            update("waiting_for_validation")
            wait_for_validation(run)
        selection_path = out / "selection.json"
        frozen = frozen_choices(run)
        if selection_path.exists():
            assert sweep.read(selection_path) == frozen
        else:
            sweep.write(selection_path, frozen)
        (out / "evaluation_source.py").write_bytes(Path(__file__).read_bytes())
        for person in sweep.PEOPLE:
            if (out / person / "test.json").exists():
                continue
            update("testing", participant=person)
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), str(run),
                            "--participant", person], cwd=sweep.ROOT, check=True)
        update("verifying")
        summarize(out, frozen)
        update("verified", summary="population_summary.json")
    except BaseException as error:
        update("failed", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_run", type=Path)
    parser.add_argument("--wait-for-validation", action="store_true")
    parser.add_argument("--participant", choices=sweep.PEOPLE, help=argparse.SUPPRESS)
    args = parser.parse_args()
    run = args.validation_run.resolve()
    if run.parent != (sweep.ROOT / "outputs").resolve() or not (run / "source_manifest.json").is_file():
        parser.error("Choose a T1 decoder preference run in outputs/")
    out = run / "heldout"
    if args.participant:
        score_participant(run, out, args.participant)
        return
    out.mkdir(exist_ok=True)
    with (out / "evaluation.log").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = out / "pipeline_status.json"
        if not status.exists() or sweep.read(status)["stage"] != "verified":
            supervise(run, out, args.wait_for_validation)


if __name__ == "__main__":
    main()

