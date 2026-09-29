"""Trace the reduced three-column targets to the original WAY-EEG-GAL sensors.

Run from the repository root with the original P1 and P6 folders beside it:
    python documentation/verify_kinematics.py

The script reads the data without modifying it and writes a compact JSON report.
"""

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


REPO = Path(__file__).resolve().parents[1]
ORIGINAL = REPO.parent
OUTPUT = REPO / "documentation" / "kinematics_verification.json"
AXES = ("x", "y", "z")
POSITION_COLUMNS = (tuple(range(18, 22)), tuple(range(22, 26)), tuple(range(26, 30)))


def report_subject(participant):
    clean_dir = REPO / "way_eeg_clean"
    clean = loadmat(clean_dir / f"kin_P{participant}.mat")["kin"]
    starts = loadmat(clean_dir / f"session_start_P{participant}.mat")["session_start"].ravel().astype(int) - 1
    if len(starts) != 9 or starts[0] != 0:
        raise ValueError(f"Unexpected P{participant} session boundaries: {starts}")

    series = []
    for number in range(1, 10):
        hs = loadmat(
            ORIGINAL / f"P{participant}" / f"HS_P{participant}_S{number}.mat",
            squeeze_me=True, struct_as_record=False,
        )["hs"]
        lo = starts[number - 1]
        hi = starts[number] if number < 9 else len(clean)
        raw = np.asarray(hs.kin.sig)
        if len(raw) != hi - lo or raw.shape[1] != 36:
            raise ValueError(f"P{participant} S{number}: original/reduced sample counts differ")
        labels = [str(hs.kin.names[i]) for i in (21, 25, 29)]
        expected = [f"P{axis}4 - position {axis} sensor 4" for axis in AXES]
        if labels != expected:
            raise ValueError(f"Unexpected kinematic channel labels: {labels}")
        series.append(raw[:, 18:30])

    raw_position = np.concatenate(series, axis=0)
    if len(raw_position) != len(clean) or clean.shape[1] != 3:
        raise ValueError(f"P{participant}: original/reduced shapes differ")

    interior = np.ones(len(clean), dtype=bool)
    for number, lo in enumerate(starts):
        hi = starts[number + 1] if number < 8 else len(clean)
        interior[lo:min(lo + 100, hi)] = False
        interior[max(hi - 100, lo):hi] = False

    axes = {}
    for axis_index, axis in enumerate(AXES):
        y = clean[:, axis_index]
        candidates = raw_position[:, POSITION_COLUMNS[axis_index][0] - 18:
                                   POSITION_COLUMNS[axis_index][-1] - 18 + 1]
        correlations = [float(np.corrcoef(candidates[:, sensor], y)[0, 1])
                        for sensor in range(4)]
        x = candidates[:, 3]
        slope, intercept = np.polyfit(x[interior], y[interior], 1)
        mean_absolute_residual = float(np.mean(
            np.abs(y[interior] - (slope * x[interior] + intercept))
        ))
        axes[axis] = {
            "reduced_column_1_based": axis_index + 1,
            "original_channel": f"P{axis}4 - position {axis} sensor 4",
            "correlation_with_original_sensors_1_to_4": correlations,
            "clean_min": float(np.min(y)), "clean_max": float(np.max(y)),
            "clean_mean": float(np.mean(y)), "clean_standard_deviation": float(np.std(y)),
            "approximate_affine_slope": float(slope),
            "approximate_affine_intercept": float(intercept),
            "interior_mean_absolute_residual": mean_absolute_residual,
        }

    return {"participant": participant, "sample_count": len(clean),
            "series_count": len(series), "all_series_lengths_match": True, "axes": axes}


def main():
    results = {
        "method": "Align the nine original HS series with the reduced kin matrix using session_start; correlate each reduced column with all four same-axis original position sensors at matching sample indices. Fit an approximate affine map to wrist values, omitting 100 samples at each series edge.",
        "limits": "High correlation and an approximate affine map identify the source channels but do not establish the exact undocumented filtering/scaling algorithm.",
        "participants": {f"P{number}": report_subject(number) for number in (1, 6)},
    }
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    for label, result in results["participants"].items():
        print(label, "samples", result["sample_count"])
        for axis, info in result["axes"].items():
            print(axis, info["original_channel"],
                  "r =", f'{info["correlation_with_original_sensors_1_to_4"][3]:.7f}',
                  "range =", (info["clean_min"], info["clean_max"]))


if __name__ == "__main__":
    main()
