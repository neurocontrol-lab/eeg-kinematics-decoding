"""Exploratory R6 population profiles using verified EXP-042 2-ms predictions."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import correlate

root = Path.cwd()
sys.path.insert(0, str(root / "src"))
from eeg_kinematics.error_dynamics import STAGES, derivatives, trial_weights
from eeg_kinematics.paths import RAW_DATA_DIR, REDUCED_DATA_DIR

EVENTS = (
    "LEDOn", "tHandStart", "tFirstDigitTouch", "tBothStartLoadPhase",
    "tLiftOff", "LEDOff", "tReplace", "tBothReleased", "tHandStop",
)
WRIST = [21, 25, 29]
MODELS = {
    "ridge": ("ridge_gap1",),
    "cnn_elu": tuple(f"cnn_elu_gap1_seed{s}" for s in (42, 43, 44)),
}
REGIONS = ("diagonal", "vertical", "horizontal", "both", "neither", "intermediate")
EDGES = np.linspace(0, 1, 51)
FS = 500
SPEED_MIN = 0.02  # Same fixed reduced-coordinate threshold as R6.
AXIS_VELOCITY_MIN = 0.02  # Deadband for signed axis motion.
TANGENTIAL_MIN = 0.05  # Deadband for speed change in reduced units / s^2.
MIN_TRIALS = 5


def raw_struct(path, name):
    return loadmat(path, squeeze_me=True, struct_as_record=False)[name]


def stage_alignment(person, series, size, all_lifts, columns):
    folder = RAW_DATA_DIR / person
    hs = raw_struct(folder / f"HS_{person}_S{series}.mat", "hs")
    ws = raw_struct(folder / f"WS_{person}_S{series}.mat", "ws")
    x = hs.eeg.sig[:, 0].astype(float)
    assert hs.eeg.samplingrate == hs.kin.samplingrate == FS
    assert len(x) == len(hs.kin.sig) == size
    rows = all_lifts[all_lifts[:, columns["Run"]] == series]
    assert len(rows) == len(ws.win)
    offsets = []
    width = 128
    for j in (0, len(rows) // 2, len(rows) - 1):
        trial = ws.win[j]
        fragment = trial.eeg[:width, 0].astype(float)
        distance = np.convolve(x*x, np.ones(width), mode="valid")
        distance -= 2 * correlate(x, fragment, mode="valid", method="fft")
        k = int(distance.argmin())
        np.testing.assert_array_equal(hs.eeg.sig[k:k+width], trial.eeg[:width])
        offsets.append(float(trial.trial_start_time - k / FS))
    assert np.ptp(offsets) < 1 / FS
    offset = np.median(offsets)
    stage = np.zeros(size, dtype=int)
    trial_ids = np.zeros(size, dtype=int)
    invalid = []
    for j, (row, trial) in enumerate(zip(rows, ws.win), start=1):
        start = row[columns["StartTime"]]
        assert abs(start - trial.trial_start_time) < 1e-9
        k = round((start - offset) * FS)
        np.testing.assert_array_equal(hs.eeg.sig[k:k+100], trial.eeg[:100])
        np.testing.assert_array_equal(hs.kin.sig[k:k+100, WRIST], trial.kin[:100, WRIST])
        event_times = np.array([start + row[columns[e]] - offset for e in EVENTS])
        if (not np.isfinite(event_times).all() or np.any(np.diff(event_times) < 0)
                or event_times[0] < 0 or event_times[-1] > size / FS):
            invalid.append(j)
            continue
        bounds = np.rint(event_times * FS).astype(int)
        assert not trial_ids[bounds[0]:bounds[-1]].any()
        for s, (lo, hi) in enumerate(zip(bounds, bounds[1:]), start=1):
            stage[lo:hi] = s
            trial_ids[lo:hi] = j
    return stage, trial_ids, {"clock_trials": len(rows), "invalid_event_trials": invalid}


def regions(observed, predicted, anchors):
    def in_anchor(value):
        bins = np.clip(np.searchsorted(EDGES, value, side="right") - 1, 0, 49)
        return np.isin(bins, anchors) & (value >= 0) & (value <= 1)

    error = np.abs(predicted - observed)
    obs_anchor, pred_anchor = in_anchor(observed), in_anchor(predicted)
    membership = np.full(len(observed), 5)
    membership[error <= 0.05] = 0
    high = error >= 0.10
    membership[high & obs_anchor & ~pred_anchor] = 1
    membership[high & ~obs_anchor & pred_anchor] = 2
    membership[high & obs_anchor & pred_anchor] = 3
    membership[high & ~obs_anchor & ~pred_anchor] = 4
    return membership, obs_anchor



def paired_slow_motion(stages, trials, streak, observed_anchor, derivatives_by_window,
                       participant, model, fit):
    """Compare slow-motion shares within the same trial and airborne stage."""
    airborne = (trials > 0) & (stages == STAGES.index("airborne_before_lowering_cue"))
    reference = {
        "same_stage": airborne & ~streak,
        "same_stage_and_anchor": airborne & ~streak & observed_anchor,
    }
    target = airborne & streak
    rows = []
    for comparator, mask in reference.items():
        paired_trials = np.intersect1d(trials[target], trials[mask])
        for trial in paired_trials:
            a = target & (trials == trial)
            b = mask & (trials == trial)
            for window, deriv in derivatives_by_window.items():
                slow = np.linalg.norm(deriv[:, 0], axis=1) <= SPEED_MIN
                rows.append(dict(participant=participant, model=model, fit=fit,
                                 comparator=comparator, window=window,
                                 series=int(trial // 1000), trial=int(trial % 1000),
                                 streak_targets=int(a.sum()), reference_targets=int(b.sum()),
                                 streak_slow=int(slow[a].sum()), reference_slow=int(slow[b].sum())))
    return rows


def main(destination):
    source = root / "outputs/timing-alignment-population-gpu-20261028T133000Z"
    for name in ("verification.json", "population_verification.json"):
        assert json.loads((source / name).read_text())["status"] == "passed"
    meta = json.loads((source / "results.json").read_text())["participant_results"]
    records, coverage, paired = [], {}, []
    for n in range(1, 13):
        person = f"P{n}"
        participant = raw_struct(RAW_DATA_DIR / person / f"{person}_AllLifts.mat", "P")
        columns = {str(name): i for i, name in enumerate(participant.ColNames)}
        kin = loadmat(REDUCED_DATA_DIR / f"kin_{person}.mat")["kin"].astype(float)
        bounds = meta[person]["series_bounds"]
        params = meta[person]["scaling"]
        center, scale = np.array(params["target_mean"]), np.array(params["target_scale"])
        with np.load(source / person / "split_indices.npz") as split:
            targets, series = split["test_targets"], split["test_series_ids"]
        stages = np.zeros(len(targets), dtype=int)
        trials = np.zeros_like(stages)
        d = {window: np.zeros((len(targets), 3, 3)) for window in (21, 41, 81)}
        for s in (8, 9):
            lo, hi = bounds[str(s)]
            stage, trial_ids, coverage_row = stage_alignment(
                person, s, hi - lo, participant.AllLifts, columns,
            )
            mask = series == s
            local = targets[mask] - lo
            assert np.all((local >= 0) & (local < hi - lo))
            stages[mask] = stage[local]
            eligible = (local >= 40) & (local < hi - lo - 40) & (trial_ids[local] > 0)
            trials[mask] = np.where(eligible, s * 1000 + trial_ids[local], 0)
            for window in d:
                d[window][mask] = derivatives(kin[lo:hi], window)[local]
            coverage_row.update(targets=int(mask.sum()), trial_targets=int(eligible.sum()),
                                contributing_trials=int(np.unique(trial_ids[local][eligible]).size))
            coverage[f"{person}/{s}"] = coverage_row
        active = trials > 0
        weights = np.zeros(len(targets))
        weights[active] = trial_weights(trials[active])
        observed = kin[targets]
        for model, fits in MODELS.items():
            for fit in fits:
                with np.load(source / person / f"{fit}_predictions.npz") as saved:
                    np.testing.assert_array_equal(saved["target_sample_index"], targets)
                    np.testing.assert_array_equal(saved["series_id"], series)
                    np.testing.assert_allclose(saved["y_true"] * scale + center,
                                               observed, rtol=1e-6, atol=1e-7)
                    predicted = saved["y_pred"] * scale + center
                for axis in range(3):
                    train = kin[:bounds["7"][0], axis]
                    anchors = np.argsort(-np.histogram(train, EDGES)[0], kind="stable")[:5]
                    labels, observed_anchor = regions(observed[:, axis], predicted[:, axis], anchors)
                    if axis == 2:
                        paired.extend(paired_slow_motion(
                            stages, trials, labels == 1, observed_anchor, d, person, model, fit,
                        ))
                    for r, label in enumerate(("all", *REGIONS)):
                        mask = active if r == 0 else active & (labels == r - 1)
                        count = np.unique(trials[mask]).size
                        for window, deriv in d.items():
                            v, a = deriv[:, 0], deriv[:, 1]
                            speed = np.linalg.norm(v, axis=1)
                            moving = speed > SPEED_MIN
                            tangential = np.zeros(len(targets))
                            tangential[moving] = np.einsum("ij,ij->i", v[moving], a[moving]) / speed[moving]
                            moving_mask = mask & moving
                            shares = {
                                "slow": speed <= SPEED_MIN,
                                "axis_positive": v[:, axis] > AXIS_VELOCITY_MIN,
                                "axis_negative": v[:, axis] < -AXIS_VELOCITY_MIN,
                                "speeding_up": tangential > TANGENTIAL_MIN,
                                "slowing_down": tangential < -TANGENTIAL_MIN,
                                **{stage: stages == j for j, stage in enumerate(STAGES) if j},
                            }
                            row = dict(participant=person, model=model, fit=fit, axis="XYZ"[axis],
                                       region=label, window=window, targets=int(mask.sum()),
                                       trials=count, moving_trials=np.unique(trials[moving_mask]).size)
                            for key, event in shares.items():
                                subset = moving_mask if key in ("speeding_up", "slowing_down") else mask
                                enough = np.unique(trials[subset]).size >= MIN_TRIALS
                                row[key] = float(np.average(event[subset], weights=weights[subset])) if enough else None
                            records.append(row)
        print(person, coverage[f"{person}/8"], coverage[f"{person}/9"], flush=True)
    destination.mkdir(exist_ok=False)
    with (destination / "seed_profiles.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    with (destination / "paired_airborne_trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=paired[0])
        writer.writeheader()
        writer.writerows(paired)
    (destination / "alignment.json").write_text(json.dumps(coverage, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))


