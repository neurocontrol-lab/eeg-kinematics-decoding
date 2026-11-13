"""Verify saved T7 predictions and summarize participant-level skill."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_position_velocity import PARTICIPANTS, SEED, file_hash, read_json

MODELS = {"ridge": "Ridge", "cnn_elu": "CNN + ELU"}
TARGETS = ("position", "velocity")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def collect(out):
    rows, verification = [], {
        "status": "passed", "experiment_id": "EXP-040",
        "participants_checked": len(PARTICIPANTS), "test_prediction_arrays_checked": 0,
        "test_metric_sets_checked": 0, "edge_labels": dict.fromkeys(("train", "validation", "test"), 0),
    }
    count_keys = ("derivative_targets_checked", "input_files_checked", "validation_fits_replayed",
                  "t1_test_replays", "velocity_test_predictions", "ridge_axis_fits_at_budget")
    for key in count_keys:
        verification[key] = 0
    selection = read_json(out / "selection.json")["participants"]
    for person in PARTICIPANTS:
        directory = out / person
        test = read_json(directory / "test.json")
        assert test["fits_sha256"] == selection[person]["fits_sha256"] == file_hash(directory / "fits.json")
        counts = test["edge_labels"]
        for key in verification["edge_labels"]:
            verification["edge_labels"][key] += counts[key]
        for key in count_keys:
            verification[key] += test[key]
        by_target = {}
        common_targets = common_ids = None
        for target in TARGETS:
            by_target[target] = {}
            target_y = None
            for name, entry in test["scores"][target].items():
                path = directory / entry["predictions"]
                assert file_hash(path) == entry["sha256"]
                with np.load(path) as data:
                    y = data["y_true"].astype(float)
                    pred = data["y_pred"].astype(float)
                    t, ids = data["target_sample_index"], data["series_id"]
                if common_targets is None:
                    common_targets, common_ids = t, ids
                np.testing.assert_array_equal(t, common_targets)
                np.testing.assert_array_equal(ids, common_ids)
                assert set(ids) == {8, 9} and len(t) == len(np.unique(t))
                if target_y is None:
                    target_y = y
                np.testing.assert_array_equal(y, target_y)
                if name == "training_mean":
                    assert not pred.any()
                error = np.square(y - pred)
                mse = error.mean(axis=0)
                total = np.square(y - y.mean(axis=0)).mean(axis=0)
                r2 = 1 - mse / total
                np.testing.assert_allclose(mse, entry["aggregate"]["mse"], atol=1e-12)
                np.testing.assert_allclose(r2, entry["aggregate"]["r2"], atol=1e-12)
                for series in (8, 9):
                    np.testing.assert_allclose(error[ids == series].mean(axis=0),
                                               entry["by_series"][str(series)]["mse"], atol=1e-12)
                verification["test_prediction_arrays_checked"] += 1
                verification["test_metric_sets_checked"] += 3
                by_target[target][name] = (mse, r2, len(y))
            baseline = by_target[target]["training_mean"][0]
            for model in MODELS:
                mse, r2, n = by_target[target][model]
                row = {"participant": person, "model": model, "target": target, "test_samples": n,
                       "ridge_alpha": selection[person][f"{target}_alpha"] if model == "ridge" else "",
                       "cnn_seed": SEED if model == "cnn_elu" else "",
                       "mse": float(mse.mean()), "mean_baseline_mse": float(baseline.mean()),
                       "skill": float(1 - mse.mean() / baseline.mean()),
                       "skill_vs_zero_velocity": float(1 - mse.mean() / by_target[target]["zero_velocity"][0].mean())
                       if target == "velocity" else ""}
                for axis, e, r in zip("xyz", mse, r2):
                    row[f"mse_{axis}"], row[f"r2_{axis}"] = float(e), float(r)
                for series in (8, 9):
                    entries = test["scores"][target]
                    m = entries[model]["by_series"][str(series)]["mse_mean"]
                    b = entries["training_mean"]["by_series"][str(series)]["mse_mean"]
                    row[f"skill_series_{series}"] = 1 - m / b
                rows.append(row)
    assert verification["test_prediction_arrays_checked"] == 84
    assert verification["validation_fits_replayed"] == 72
    assert verification["t1_test_replays"] == verification["velocity_test_predictions"] == 24
    return rows, verification


def report(out):
    rows, verified = collect(out)
    with (out / "participant_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    draws = np.random.default_rng(20261110).integers(0, len(PARTICIPANTS), (10000, len(PARTICIPANTS)))

    def interval(x):
        return np.quantile(x[draws].mean(axis=1), [.025, .975]).tolist()

    summaries = {}
    lines = [
        "# EXP-040 / T7 — Wrist position and velocity decoding", "",
        "P1–P12, separately fitted within each participant. The verified T1 2 ms position fits "
        "and new velocity fits use identical EEG windows, timestamps and series splits. "
        "Each model is compared with the training mean of its own target. Positive skill means "
        "the model beats that reference.", "",
        "## Held-out performance on series 8–9", "",
        "| Model | Target | Mean skill | Exploratory 95% interval | Beat training mean | Mean R² X / Y / Z |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, layout="constrained")
    for (model, label), ax in zip(MODELS.items(), axes):
        skills = {}
        for target in TARGETS:
            picked = [r for r in rows if r["model"] == model and r["target"] == target]
            assert [r["participant"] for r in picked] == list(PARTICIPANTS)
            values = np.array([r["skill"] for r in picked])
            skills[target] = values
            low, high = interval(values)
            axis_r2 = [float(np.mean([r[f"r2_{a}"] for r in picked])) for a in "xyz"]
            key = f"{model}_{target}"
            summaries[key] = {
                "mean_skill": float(values.mean()), "interval": [low, high],
                "beat_training_mean": int((values > 0).sum()), "axis_r2": axis_r2,
                "mean_test_mse": float(np.mean([r["mse"] for r in picked])),
                "mean_train_mean_baseline_mse": float(np.mean([r["mean_baseline_mse"] for r in picked])),
                "mean_skill_by_series": [float(np.mean([r[f"skill_series_{s}"] for r in picked])) for s in (8, 9)],
            }
            if target == "velocity":
                v = np.array([r["skill_vs_zero_velocity"] for r in picked])
                summaries[key]["zero_velocity_comparison"] = {
                    "mean_skill": float(v.mean()), "interval": interval(v), "count_positive": int((v > 0).sum())
                }
            lines.append(f"| {label} | {target.capitalize()} | {values.mean():.3f} | [{low:.3f}, {high:.3f}] | "
                         f"{(values > 0).sum()}/12 | " + " / ".join(f"{v:.3f}" for v in axis_r2) + " |")
        delta = skills["velocity"] - skills["position"]
        summaries[f"{model}_velocity_minus_position"] = {
            "mean_delta": float(delta.mean()), "interval": interval(delta),
            "velocity_higher_skill": int((delta > 0).sum()),
        }
        for first, second in zip(skills["position"], skills["velocity"]):
            ax.plot((0, 1), (first, second), marker="o", color="#658098", alpha=.55)
        for i, target in enumerate(TARGETS):
            values = skills[target]
            low, high = interval(values)
            avg = values.mean()
            ax.errorbar(i, avg, yerr=[[avg-low], [high-avg]], fmt="D", color="#b04e25", capsize=4, zorder=10)
        ax.axhline(0, color="#747a85", linewidth=1, linestyle="--")
        ax.set(xticks=(0, 1), xticklabels=("Position", "Velocity"), xlim=(-.28, 1.28), title=label)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Test MSE skill versus target training mean")
    fig.suptitle("T7 | Wrist position and velocity on matched EEG windows\n"
                 "Lines pair participants. Diamonds show means and exploratory 95% intervals.")
    fig.savefig(out / "population_target_skill.png", dpi=170)
    plt.close(fig)

    lines += ["", "![Paired participant scores](population_target_skill.png)", "",
              "## Paired differences and zero-velocity reference", ""]
    for model, label in MODELS.items():
        d = summaries[f"{model}_velocity_minus_position"]
        lo, hi = d["interval"]
        z = summaries[f"{model}_velocity"]["zero_velocity_comparison"]
        lines.append(f"- **{label}.** Mean velocity-minus-position skill {d['mean_delta']:+.3f} "
                     f"(95% [{lo:+.3f}, {hi:+.3f}]). Velocity had higher skill in "
                     f"{d['velocity_higher_skill']}/12 participants. Against zero velocity, "
                     f"mean skill was {z['mean_skill']:.3f} and {z['count_positive']}/12 beat the reference.")
    edges = verified["edge_labels"]
    lines += [
        "", "## Scope", "",
        "The velocity label is the signed derivative of the wrist trace. It uses a 41-sample cubic "
        "Savitzky–Golay filter evaluated separately within each recording series. Its interior "
        "support includes 40 ms of observed kinematics on either side of the target time. "
        f"Series-local edge evaluation was used for {edges['train']} training, "
        f"{edges['validation']} validation and {edges['test']} test target times.",
        "",
        "Each participant has equal weight in the means and bootstrap. Both targets were scaled from "
        "their own training series. Skill values compare performance relative to different targets’ "
        "training-mean references. Differences in signal dynamics and label noise also affect these "
        "scores. The analysis uses one CNN seed and one derivative definition on familiar held-out series.",
        "", "## Verification", "",
        f"Rebuilt {verified['derivative_targets_checked']:,} velocity target times, "
        f"replayed {verified['validation_fits_replayed']} velocity validation fits and "
        f"{verified['t1_test_replays']} reused position test fits, and checked "
        f"{verified['test_metric_sets_checked']} test metric sets. "
        f"{verified['ridge_axis_fits_at_budget']} velocity ridge axis fits reached the iteration cap.",
        "", "[Declared protocol](declared_protocol.md) · [Participant metrics](participant_metrics.csv) · "
        "[Verification](verification.json) · [Population summary](population_summary.json)", "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "population_summary.json", summaries)
    write_json(out / "verification.json", verified)


if __name__ == "__main__":
    report(Path(sys.argv[1]))

