"""Run T7 with the verified T1 2-ms position reference and matched velocity fits."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key, value in {
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    "OPENBLAS_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "4",
    "MPLBACKEND": "Agg",
}.items():
    os.environ.setdefault(key, value)

ROOT = Path(__file__).resolve().parents[2]
TIMING = ROOT / "experiments/07_timing_alignment"
REFERENCE = ROOT / "outputs/timing-alignment-population-gpu-20261028T133000Z"
PROTOCOL = ROOT / "documentation/methods/POSITION_VELOCITY.md"
REPORT = Path(__file__).with_name("report_position_velocity.py")
SOURCE_FILES = {
    "runner_source.py": Path(__file__),
    "report_source.py": REPORT,
    "declared_protocol.md": PROTOCOL,
    "t1_runner_source.py": TIMING / "run_population_timing.py",
    "cnn_reference_source.py": TIMING / "run_timing_alignment.py",
    "window_reference_source.py": ROOT / "src/eeg_kinematics/timing_alignment.py",
    "paths_source.py": ROOT / "src/eeg_kinematics/paths.py",
    "t1_verification.json": REFERENCE / "verification.json",
    "t1_population_verification.json": REFERENCE / "population_verification.json",
    "t1_results.json": REFERENCE / "results.json",
}
PARTICIPANTS = tuple(f"P{i}" for i in range(1, 13))
ALPHAS = (1., 10., 100., 1000., 10000.)
SEED = 42
WINDOW = 41
FS = 500


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_npz(path, **arrays):
    import numpy as np

    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def set_up_worker():
    sys.path[:0] = [str(TIMING), str(ROOT / "src")]
    import run_population_timing as t1

    tf = t1.tf
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("A TensorFlow GPU is required for CNN + ELU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    with tf.device("/GPU:0"):
        probe = tf.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
    assert "GPU:0" in probe.device
    probe.numpy()
    return t1, tf.config.experimental.get_device_details(devices[0])


def load_data(t1, person):
    import numpy as np
    from scipy.signal import savgol_filter

    original = read_json(REFERENCE / "results.json")["participant_results"][person]
    assert all(t1.sha(t1.REDUCED_DATA_DIR / name) == expected
               for name, expected in original["input_sha256"].items())
    eeg, position, bounds, splits, scaling = t1.dataset(person)
    assert scaling == original["scaling"]
    assert {str(k): list(b) for k, b in bounds.items()} == original["series_bounds"]
    with np.load(REFERENCE / person / "split_indices.npz") as saved:
        for name, (indices, series) in splits.items():
            np.testing.assert_array_equal(indices, saved[f"{name}_targets"])
            np.testing.assert_array_equal(series, saved[f"{name}_series_ids"])
    kin = t1.loadmat(t1.REDUCED_DATA_DIR / f"kin_{person}.mat")["kin"].astype(np.float64)
    assert kin.shape == position.shape
    velocity = np.empty_like(kin)
    for lo, hi in bounds.values():
        velocity[lo:hi] = savgol_filter(
            kin[lo:hi], WINDOW, 3, deriv=1, delta=1 / FS, axis=0, mode="interp",
        )
    assert np.isfinite(velocity).all()
    scaler = t1.StandardScaler()
    for series in t1.SPLITS["train"]:
        lo, hi = bounds[series]
        scaler.partial_fit(velocity[lo:hi])
    zero = scaler.transform(np.zeros((1, 3)))[0]
    velocity_scaling = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                        "samples_seen": int(scaler.n_samples_seen_),
                        "fit_series": list(t1.SPLITS["train"])}
    assert velocity_scaling["samples_seen"] == scaling["samples_seen"]
    assert np.all(scaler.var_ > 0)
    return original, eeg, position, kin, velocity, scaler, zero, bounds, splits, velocity_scaling


def fit(t1, out, person, gpu):
    import numpy as np

    original, eeg, _, _, velocity, scaler, _, bounds, splits, scaling = load_data(t1, person)
    directory = out / person
    directory.mkdir(exist_ok=True)
    checkpoint = directory / "fits.json"
    versions = {"tensorflow": t1.tf.__version__, "numpy": np.__version__,
                "scipy": t1.scipy.__version__, "sklearn": t1.sklearn.__version__}
    state = read_json(checkpoint) if checkpoint.is_file() else {
        "ridge": {}, "velocity_scaler": scaling, "runtime_versions": versions,
        "gpu": gpu, "input_sha256": original["input_sha256"],
    }
    assert state["velocity_scaler"] == scaling
    assert state["runtime_versions"] == versions
    assert state["input_sha256"] == original["input_sha256"]
    train = t1.make_windows(eeg, *splits["train"], bounds, 1)
    val = t1.make_windows(eeg, *splits["validation"], bounds, 1)
    yt = scaler.transform(velocity[splits["train"][0]]).astype(np.float32)
    yv = scaler.transform(velocity[splits["validation"][0]]).astype(np.float32)
    x_train, x_val = train.reshape(len(train), -1), val.reshape(len(val), -1)
    for alpha in ALPHAS:
        key = str(alpha)
        if key in state["ridge"]:
            entry = state["ridge"][key]
            assert t1.sha(directory / entry["weights"]) == entry["sha256"]
            continue
        begin = time.monotonic()
        model = t1.Ridge(alpha=alpha, solver="lsqr", tol=.001, max_iter=200).fit(x_train, yt)
        pred = model.predict(x_val)
        weights = directory / f"velocity_ridge_alpha{alpha:g}.npz"
        save_npz(weights, coef=model.coef_, intercept=model.intercept_, validation_pred=pred)
        state["ridge"][key] = {
            "alpha": alpha, "weights": weights.name, "sha256": t1.sha(weights),
            "n_iter": model.n_iter_.tolist(), "fit_seconds": time.monotonic() - begin,
            "validation": t1.metrics(yv, pred),
        }
        t1.save_json(checkpoint, state)
        print("RIDGE_DONE", person, alpha, state["ridge"][key]["validation"]["mse_mean"], flush=True)

    if "cnn" not in state:
        tf = t1.tf
        tf.keras.utils.set_random_seed(SEED)
        print("CNN_START", person, "velocity", SEED, flush=True)
        begin = time.monotonic()
        with tf.device("/GPU:0"):
            model = t1.build_cnn()
            init = hashlib.sha256(b"".join(w.tobytes() for w in model.get_weights())).hexdigest()
            assert init == original["gaps"]["1"]["cnn_elu"][str(SEED)]["initial_sha256"]
            history = model.fit(train[..., None], yt, validation_data=(val[..., None], yv),
                                epochs=15, batch_size=64, verbose=0)
            pred = model.predict(val[..., None], batch_size=64, verbose=0)
            temporary = directory / "velocity_cnn.tmp.keras"
            model.save(temporary)
        weights = directory / "velocity_cnn.keras"
        temporary.replace(weights)
        validation = directory / "velocity_cnn_validation.npz"
        save_npz(validation, y_pred=pred)
        state["cnn"] = {
            "weights": weights.name, "sha256": t1.sha(weights), "seed": SEED,
            "initial_sha256": init, "parameters": model.count_params(),
            "validation_file": validation.name, "validation_sha256": t1.sha(validation),
            "history": history.history, "validation": t1.metrics(yv, pred),
            "fit_seconds": time.monotonic() - begin,
        }
        t1.save_json(checkpoint, state)
        print("CNN_DONE", person, state["cnn"]["fit_seconds"], flush=True)


def replay_ridge(path, x):
    import numpy as np

    with np.load(path) as saved:
        return x.reshape(len(x), -1) @ saved["coef"].T + saved["intercept"]


def replay_cnn(tf, path, x):
    tf.keras.backend.clear_session()
    with tf.device("/GPU:0"):
        model = tf.keras.models.load_model(path, compile=False)
        return model.predict(x[..., None], batch_size=64, verbose=0)


def verify_derivative(kin, velocity, bounds, splits):
    import numpy as np
    from scipy.signal import savgol_coeffs

    half = WINDOW // 2
    count = edge_count = 0
    edges_by_split = {}
    coefficients = np.stack([savgol_coeffs(WINDOW, 3, deriv=1, delta=1 / FS,
                                          pos=i, use="dot") for i in range(WINDOW)])
    offsets = np.arange(WINDOW)
    for split, (targets, ids) in splits.items():
        edges_before = edge_count
        for series in np.unique(ids):
            t = targets[ids == series]
            lo, hi = bounds[int(series)]
            start = np.clip(t - half, lo, hi - WINDOW)
            pos = t - start
            fit = np.einsum("ij,ijk->ik", coefficients[pos], kin[start[:, None] + offsets])
            np.testing.assert_allclose(fit, velocity[t], atol=1e-8, rtol=1e-8)
            count += len(t)
            edge_count += np.count_nonzero(pos != half)
        edges_by_split[split] = int(edge_count - edges_before)
    return count, edges_by_split


def evaluate(t1, out, person):
    import numpy as np

    frozen = read_json(out / "selection.json")["participants"][person]
    directory = out / person
    state = read_json(directory / "fits.json")
    assert frozen["fits_sha256"] == t1.sha(directory / "fits.json")
    original, eeg, position, kin, velocity, scaler, zero, bounds, splits, scaling = load_data(t1, person)
    assert scaling == state["velocity_scaler"]
    derivative_count, edges = verify_derivative(kin, velocity, bounds, splits)
    velocity_scaled = {name: scaler.transform(velocity[t]).astype(np.float32)
                       for name, (t, _) in splits.items()}
    val = t1.make_windows(eeg, *splits["validation"], bounds, 1)
    y_val = velocity_scaled["validation"]
    selected = []
    for key, entry in state["ridge"].items():
        path = directory / entry["weights"]
        assert entry["sha256"] == t1.sha(path)
        replay = replay_ridge(path, val)
        with np.load(path) as saved:
            np.testing.assert_allclose(replay, saved["validation_pred"], atol=1e-6, rtol=1e-6)
        mse = t1.metrics(y_val, replay)["mse_mean"]
        np.testing.assert_allclose(mse, entry["validation"]["mse_mean"], atol=1e-9)
        selected.append((mse, entry["alpha"], key))
    assert min(selected)[1] == frozen["velocity_alpha"]
    cnn_entry = state["cnn"]
    cnn = directory / cnn_entry["weights"]
    val_file = directory / cnn_entry["validation_file"]
    assert t1.sha(cnn) == cnn_entry["sha256"] and t1.sha(val_file) == cnn_entry["validation_sha256"]
    replay = replay_cnn(t1.tf, cnn, val)
    with np.load(val_file) as saved:
        np.testing.assert_allclose(replay, saved["y_pred"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(t1.metrics(y_val, replay)["mse_mean"],
                               cnn_entry["validation"]["mse_mean"], atol=1e-6)
    del val

    indices, ids = splits["test"]
    x = t1.make_windows(eeg, indices, ids, bounds, 1)
    position_entries = original["gaps"]["1"]
    pos_ridge = next(c for c in position_entries["ridge"]["candidates"]
                     if c["alpha"] == frozen["position_alpha"])
    pos_cnn = position_entries["cnn_elu"][str(SEED)]
    assert pos_ridge["alpha"] == position_entries["ridge"]["selected_alpha"]
    assert cnn_entry["initial_sha256"] == pos_cnn["initial_sha256"]
    for entry in (pos_ridge, pos_cnn):
        assert t1.sha(REFERENCE / person / entry["weights"]) == entry["weights_sha256"]
    position_preds = {
        "ridge": replay_ridge(REFERENCE / person / pos_ridge["weights"], x),
        "cnn_elu": replay_cnn(t1.tf, REFERENCE / person / pos_cnn["weights"], x),
    }
    y_position = position[indices]
    for model, pred in position_preds.items():
        key = "ridge_gap1" if model == "ridge" else "cnn_elu_gap1_seed42"
        with np.load(REFERENCE / person / f"{key}_predictions.npz") as saved:
            np.testing.assert_array_equal(indices, saved["target_sample_index"])
            np.testing.assert_array_equal(ids, saved["series_id"])
            np.testing.assert_array_equal(y_position, saved["y_true"])
            np.testing.assert_allclose(pred, saved["y_pred"], atol=3e-5, rtol=3e-5)
            position_preds[model] = saved["y_pred"]
    velocity_ridge = state["ridge"][str(frozen["velocity_alpha"])]
    velocity_preds = {
        "ridge": replay_ridge(directory / velocity_ridge["weights"], x),
        "cnn_elu": replay_cnn(t1.tf, cnn, x),
    }
    scores = {}
    for target, y, predictions in (
        ("position", y_position, position_preds),
        ("velocity", velocity_scaled["test"], velocity_preds),
    ):
        fits = {**predictions, "training_mean": np.zeros_like(y)}
        if target == "velocity":
            fits["zero_velocity"] = np.broadcast_to(zero, y.shape)
        scores[target] = {}
        for label, pred in fits.items():
            name = directory / f"{target}_{label}_test.npz"
            save_npz(name, y_true=y, y_pred=pred, target_sample_index=indices, series_id=ids)
            with np.load(name) as saved:
                record = t1.scored(saved["y_true"], saved["y_pred"], saved["series_id"])
            scores[target][label] = {**record, "predictions": name.name, "sha256": t1.sha(name)}
    t1.save_json(directory / "test.json", {
        "scores": scores, "edge_labels": edges, "derivative_targets_checked": derivative_count,
        "input_files_checked": len(original["input_sha256"]), "validation_fits_replayed": 6,
        "t1_test_replays": 2, "velocity_test_predictions": 2,
        "ridge_axis_fits_at_budget": sum(n == 200 for e in state["ridge"].values() for n in e["n_iter"]),
        "fits_sha256": frozen["fits_sha256"],
    })
    print("TEST_AND_REPLAY_DONE", person, "n_test", len(indices), "edge_labels", edges, flush=True)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_all(out):
    from datetime import datetime, timezone

    def stamp():
        return datetime.now(timezone.utc).isoformat()

    def write(path, data):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    source_hashes = {name: file_hash(path) for name, path in SOURCE_FILES.items()}
    source_manifest = out / "source_manifest.json"
    if source_manifest.exists():
        assert read_json(source_manifest) == source_hashes
        assert all(file_hash(out / name) == expected for name, expected in source_hashes.items())
    else:
        for name, path in SOURCE_FILES.items():
            (out / name).write_bytes(path.read_bytes())
        write(source_manifest, source_hashes)
    for name in ("verification.json", "population_verification.json"):
        assert read_json(REFERENCE / name)["status"] == "passed"
    t1_hashes = read_json(REFERENCE / "results.json")["source_sha256"]
    for key, name in (("runner", "t1_runner_source.py"), ("pilot_runner", "cnn_reference_source.py"),
                      ("timing_helper", "window_reference_source.py")):
        assert t1_hashes[key] == source_hashes[name]

    status = out / "pipeline_status.json"

    def state(stage, **detail):
        write(status, {"experiment_id": "EXP-040", "stage": stage, "pid": os.getpid(),
                       "updated_at": stamp(), **detail})

    def worker(phase, person):
        state(phase, participant=person)
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--output-dir",
                        str(out), "--worker", phase, person], cwd=ROOT, check=True)

    try:
        reference = read_json(REFERENCE / "results.json")["participant_results"]
        selected = {}
        for person in PARTICIPANTS:
            path = out / person / "fits.json"
            if not path.is_file() or "cnn" not in read_json(path):
                worker("fitting", person)
            fitted = read_json(path)
            assert set(fitted["ridge"]) == set(map(str, ALPHAS))
            best = min(fitted["ridge"].values(), key=lambda e: (e["validation"]["mse_mean"], e["alpha"]))
            selected[person] = {
                "velocity_alpha": best["alpha"], "fits_sha256": file_hash(path),
                "position_alpha": reference[person]["gaps"]["1"]["ridge"]["selected_alpha"],
            }
        path = out / "selection.json"
        if path.exists():
            assert read_json(path)["participants"] == selected
        else:
            write(path, {"frozen_at": stamp(), "participants": selected, "cnn_seed": SEED})
        print("ALL_VALIDATION_CHOICES_FROZEN", flush=True)
        for person in PARTICIPANTS:
            path = out / person / "test.json"
            if not path.is_file():
                worker("test_and_verify", person)
            checked = read_json(path)
            assert checked["fits_sha256"] == selected[person]["fits_sha256"]
            for models in checked["scores"].values():
                for entry in models.values():
                    assert file_hash(out / person / entry["predictions"]) == entry["sha256"]
        assert {name: file_hash(path) for name, path in SOURCE_FILES.items()} == source_hashes
        state("reporting")
        subprocess.run([sys.executable, str(REPORT), str(out)], cwd=ROOT, check=True)
        assert read_json(out / "verification.json")["status"] == "passed"
        state("verified", participants=12, report="RESULTS.md")
        print("T7_VERIFIED", out, flush=True)
    except BaseException as error:
        state("failed", last_phase=read_json(status)["stage"] if status.exists() else "setup",
              error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.parent != (ROOT / "outputs").resolve() or not out.name.startswith("position-velocity-"):
        parser.error("Use a new position-velocity-* folder directly under outputs/")
    if args.worker:
        phase, person = args.worker
        assert person in PARTICIPANTS and phase in ("fitting", "test_and_verify")
        t1, gpu = set_up_worker()
        if phase == "fitting":
            fit(t1, out, person, gpu)
        else:
            evaluate(t1, out, person)
        return
    out.mkdir(exist_ok=True)
    with (out / "training.log").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "pipeline_status.json").is_file() and read_json(out / "pipeline_status.json")["stage"] == "verified":
            print("Already verified", out)
        else:
            run_all(out)


if __name__ == "__main__":
    main()


