"""Average seed-level R6 profiles within each participant before group summaries."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(sys.argv[1])
DEST = Path(sys.argv[2])
TIMING = ROOT / "outputs/timing-alignment-population-gpu-20261028T133000Z"
PEOPLE = tuple(f"P{i}" for i in range(1, 13))
SEEDS = {"cnn_elu": 3, "ridge": 1}
GAPS = (2, 50, 100, 200, 500, 1000)
LABELS = {"ridge": "Ridge", "cnn_elu": "CNN + ELU"}
COLORS = {"ridge": "#29588f", "cnn_elu": "#ba501d"}
METRICS = (
    "slow", "axis_positive", "axis_negative", "speeding_up", "slowing_down",
    "preparation", "reach", "preload", "loading", "airborne_before_lowering_cue",
    "after_lowering_cue", "release", "return",
)


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def interval(values):
    rng = np.random.default_rng(42)
    values = np.asarray(values)
    boot = np.sort(rng.choice(values, size=(5000, len(values))).mean(axis=1))
    return float(boot[125]), float(boot[4874])


def aggregate():
    seed_groups = defaultdict(list)
    for row in read_csv(DEST / "seed_profiles.csv"):
        key = tuple(row[name] for name in ("model", "axis", "region", "window", "participant"))
        seed_groups[key].append(row)
    participant = {}
    for key, rows in seed_groups.items():
        assert len(rows) == len({r["fit"] for r in rows}) == SEEDS[key[0]]
        participant[key] = {
            metric: mean(float(r[metric]) for r in rows) if all(r[metric] for r in rows) else None
            for metric in METRICS
        }
    summaries = {}
    for model, axis, region, window, person in participant:
        key = model, axis, region, window
        if key in summaries:
            continue
        measures = {}
        for metric in METRICS:
            pairs = [
                (participant[(*key, p)][metric], participant[(model, axis, "all", window, p)][metric])
                for p in PEOPLE
            ]
            pairs = [(r, b) for r, b in pairs if r is not None and b is not None]
            if not pairs:
                continue
            changes = [100 * (r - b) for r, b in pairs]
            low, high = interval(changes)
            measures[metric] = {
                "participants": len(pairs), "share_pct": 100 * mean(r for r, _ in pairs),
                "reference_pct": 100 * mean(b for _, b in pairs), "difference_pp": mean(changes),
                "difference_95_low_pp": low, "difference_95_high_pp": high,
                "positive_count": sum(d > 1e-10 for d in changes),
                "negative_count": sum(d < -1e-10 for d in changes),
            }
        summaries[key] = measures

    rows = [dict(model=k[0], axis=k[1], region=k[2], window_samples=k[3], metric=metric, **stats)
            for k, measures in summaries.items() for metric, stats in measures.items()]
    with (DEST / "population_profiles.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return summaries



def t1_validation():
    cells = defaultdict(list)
    for row in read_csv(TIMING / "validation_candidates.csv"):
        if row["model"] == "ridge" and row["selected_alpha_at_gap"] != "True":
            continue
        key = row["model"], row["participant"], int(row["gap_ms"])
        cells[key].append(float(row["validation_mse"]))
    models = {}
    for model in SEEDS:
        changes = []
        for gap in GAPS:
            relative = []
            for p in PEOPLE:
                a, b = cells[model, p, gap], cells[model, p, 2]
                assert len(a) == len(b) == SEEDS[model]
                relative.append(100 * (mean(a) / mean(b) - 1))
            changes.append(relative)
        models[model] = np.array(changes)
    return models


def paired_slow_motion():
    rows = read_csv(DEST / "paired_airborne_trials.csv")
    groups = defaultdict(list)
    identifiers = set()
    for row in rows:
        key = tuple(row[k] for k in ("model", "comparator", "window", "participant", "fit"))
        identity = (*key, row["series"], row["trial"])
        assert identity not in identifiers
        identifiers.add(identity)
        numerator_a, numerator_b = int(row["streak_slow"]), int(row["reference_slow"])
        denominator_a, denominator_b = int(row["streak_targets"]), int(row["reference_targets"])
        assert 0 <= numerator_a <= denominator_a and 0 <= numerator_b <= denominator_b
        assert denominator_a and denominator_b
        groups[key].append((numerator_a / denominator_a, numerator_b / denominator_b))

    participants = defaultdict(list)
    for key, values in groups.items():
        if len(values) >= 5:
            a, b = np.mean(values, axis=0)
            participants[key[:4]].append((a, b, len(values)))

    participant_rows = []
    for (model, comparator, window, person), fits in participants.items():
        if len(fits) != SEEDS[model]:
            continue
        a, b, _ = np.mean(fits, axis=0)
        participant_rows.append(dict(model=model, comparator=comparator, window=window,
                                     participant=person, min_paired_trials=min(t[2] for t in fits),
                                     streak_pct=100*a, reference_pct=100*b, delta_pp=100*(a-b)))
    with (DEST / "paired_airborne_participants.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=participant_rows[0])
        writer.writeheader()
        writer.writerows(participant_rows)
    return participant_rows


def draw_t1(models):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for ax, limit, title in zip(axes, (6, 3), ("All six gaps", "Detail from 2 to 100 ms")):
        for model, effects in models.items():
            y = effects.mean(axis=1)[:limit]
            ax.plot(np.arange(limit), y, "o-", lw=2.3, label=LABELS[model], color=COLORS[model])
            if limit == 3:
                for index, value in enumerate(y):
                    if index:
                        vertical_offset = 9 if model == "ridge" else -15
                        ax.annotate(f"{value:+.2f}%", (index, value),
                                    textcoords="offset points", xytext=(0, vertical_offset),
                                    ha="center", color=COLORS[model], fontsize=9)
        ax.set(xticks=np.arange(limit), xticklabels=GAPS[:limit], xlabel="Tested EEG end gaps (ms, evenly spaced on chart)",
               ylabel="Mean change in validation MSE (%)", title=title, xlim=(-0.25, limit-0.65))
        ax.axhline(0, color="#6b7280", lw=0.9, linestyle="--")
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].set_ylim(-1.1, 2.05)
    fig.suptitle("T1 | How does validation error change with the EEG end gap?",
                 fontweight="bold", fontsize=15)
    fig.savefig(DEST / "t1_validation_gap_curve.png", dpi=165)
    plt.close(fig)


def draw_r6(summaries, paired):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": [1.55, 1]}, layout="constrained")
    stages = METRICS[5:]
    entries = [(model, axis, streak) for model in ("ridge", "cnn_elu")
               for streak in ("vertical", "horizontal") for axis in "XYZ"]
    profile = np.array([[summaries[(*key, "41")][stage]["difference_pp"] for stage in stages]
                        for key in entries])
    coverage = np.array([[summaries[(*key, "41")][stage]["participants"] for stage in stages]
                         for key in entries])
    ax = axes[0]
    h = ax.imshow(profile, aspect="auto", cmap="RdBu_r", vmin=-45, vmax=45)
    for y, row in enumerate(profile):
        for x, value in enumerate(row):
            label = f"{round(value):+d}" if round(value) else "0"
            ax.text(x, y, label, ha="center", va="center",
                    color="white" if abs(value) > 27 else "#172033", fontsize=8.4)
    names = [f"{LABELS[m]} | {axis} {r}" + (f" (n={coverage[i,0]})" if coverage[i,0] < 12 else "")
             for i, (m, axis, r) in enumerate(entries)]
    ax.set(yticks=range(len(entries)), yticklabels=names, xticks=range(len(stages)),
           xticklabels=["Prep.", "Reach", "Preload", "Loading", "Airborne\nbefore cue",
                        "After cue", "Release", "Return"],
           title="Which stages are overrepresented in each streak?")
    ax.tick_params(axis="both", length=0)
    for y in (2.5, 5.5, 8.5):
        ax.axhline(y, color="white", lw=2)
    fig.colorbar(h, ax=ax, orientation="horizontal", shrink=.82, pad=.08,
                 label="Difference from all trial targets (percentage points)")

    ax = axes[1]
    ax.axvline(0, color="#6b7280", lw=1, linestyle="--")
    for index, model in enumerate(("ridge", "cnn_elu")):
        rows = [r for r in paired if r["model"] == model and r["window"] == "41"
                and r["comparator"] == "same_stage_and_anchor"]
        y = np.array([PEOPLE.index(r["participant"]) for r in rows])
        offset = (index - .5) * .24
        values = [r["delta_pp"] for r in rows]
        ax.scatter(values, y + offset, s=27, label=f"{LABELS[model]} (n={len(rows)})",
                   color=COLORS[model], alpha=.82, zorder=2)
        low, high = interval(values)
        group = mean(values)
        ax.errorbar(group, 12 + offset, xerr=[[group-low], [high-group]], fmt="D",
                    capsize=3, color=COLORS[model], markersize=5)
    ax.set(yticks=np.arange(13), yticklabels=(*PEOPLE, "Mean"),
           xlabel="Slow-motion share in streak minus reference (pp)",
           title="Z vertical streak within the same airborne trial")
    ax.invert_yaxis()
    ax.axhline(11.5, color="#b0b7c0", lw=.8)
    ax.grid(axis="x", alpha=.18)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("R6 | Movement regimes in high-error scatter streaks",
                 fontweight="bold", fontsize=15)
    fig.savefig(DEST / "r6_population_streaks.png", dpi=170)
    plt.close(fig)


def report(models, summaries, paired):
    alignment = json.loads((DEST / "alignment.json").read_text())
    invalid = sum(len(r["invalid_event_trials"]) for r in alignment.values())
    assert sum(r["clock_trials"] for r in alignment.values()) == 816
    assert sum(r["contributing_trials"] for r in alignment.values()) == 804
    assert sum(r["trial_targets"] for r in alignment.values()) == 36811
    lines = [
        "# T1 and R6 population follow-up", "",
        "## T1 validation error at each end gap", "",
        "Mean of each participant’s percentage change from their own 2 ms validation MSE. "
        "Ridge uses the validation-selected penalty at each gap. CNN + ELU averages the "
        "three seed MSEs within participant before computing relative change. Positive values mean higher error.", "",
        "| Model | 2 ms | 50 ms | 100 ms | 200 ms | 500 ms | 1,000 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in ("ridge", "cnn_elu"):
        means = [f"{v:+.2f}%" for v in models[model].mean(axis=1)]
        lines.append("| " + " | ".join([LABELS[model], *means]) + " |")
    lines += [
        "", "![Population timing curve](t1_validation_gap_curve.png)", "",
        "The verified held-out comparison concerns a different decision. Selecting one gap "
        "per participant on validation reduced CNN + ELU mean test MSE from 0.7997 to 0.7723 "
        "(−3.42%). Ridge mean test MSE rose from 0.8970 to 0.9233 (+2.93%). All six gaps were "
        "evaluated on validation. Held-out scoring covered the selected gap and 2 ms reference.",
        "", "## R6 movement regimes in the scatter streaks", "",
        "P1–P12, series 8–9, verified T1 2 ms ridge and CNN + ELU predictions on identical target times. "
        f"All 816 trial clocks aligned. {invalid} trials were excluded for invalid event order. "
        "There were 36,811 eligible event-assigned targets in 804 trials. The original P1 R6 lead "
        "decoder was CNN–BiLSTM.", "",
        "A vertical streak is high error with only the observed position in a frequent training position "
        "bin. A horizontal streak is high error with only the predicted position in such a bin. "
        "The original R6 region rules were reused. Stage enrichment compares the share of each stage "
        "in the streak with its share among all eligible trial targets for the same participants. "
        "Trials have equal starting weight. CNN seed shares are averaged before giving each participant "
        "equal weight. The profile CSVs give direction, speeding-up and slowing-down results, "
        "as well as slow motion and stage composition.", "",
        "![Stage enrichment and matched Z-streak comparison](r6_population_streaks.png)", "",
        "### Does the Z-axis slow-motion association persist within a stage?", "",
        "The primary comparison pairs Z vertical-streak targets with other targets from the same trial, "
        "same airborne stage (lift-off to lowering cue), and same frequent-observed-bin category. "
        "The wider reference uses all other targets from that trial and stage. "
        "A trial must contain both groups. Each seed requires at least five paired trials for a participant "
        "to contribute. Trial shares are averaged per fit, then across seeds within participant, "
        "then across participants. Positive values mean the streak contains more slow motion. "
        "The intervals resample participants and are exploratory.", "",
        "| Model | Reference | Smoothing | Participants | Streak slow | Reference slow | Difference (pp) | 95% interval (pp) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for comparator in ("same_stage_and_anchor", "same_stage"):
        for model in ("ridge", "cnn_elu"):
            for window in ("41", "21", "81"):
                selected = [r for r in paired if (r["model"], r["comparator"], r["window"])
                            == (model, comparator, window)]
                deltas = [r["delta_pp"] for r in selected]
                low, high = interval(deltas)
                reference = "Same stage and frequent-position category" if comparator.endswith("anchor") else "Same stage"
                lines.append(f"| {LABELS[model]} | {reference} | {2*int(window)} ms | {len(selected)} | "
                             f"{mean(r['streak_pct'] for r in selected):.1f}% | "
                             f"{mean(r['reference_pct'] for r in selected):.1f}% | "
                             f"{mean(deltas):+.1f} | [{low:+.1f}, {high:+.1f}] |")
    lines += [
        "", "Across all trial stages, slow motion was 11.5 percentage points more common in the "
        "CNN + ELU Z vertical streak than among all eligible targets, and 12.5 points more common "
        "for ridge. For both decoders, the airborne stage before the lowering cue was also "
        "overrepresented by more than 20 points. Within the same airborne trial and frequent-position "
        "category, the estimated excess was +2.3 points for CNN + ELU and +6.3 for ridge. "
        "Both primary intervals include zero. The overall enrichment and the matched contrasts "
        "summarize different target subsets.", "",
        "", "The matched comparison adjusts for trial, stage and membership in a frequent observed-position "
        "bin. Individual bin identity and position within a bin can still differ. The geometry is defined "
        "using prediction error, and the associations are descriptive. Speed is measured from symmetric "
        "smoothing of observed kinematics in reduced coordinates. Slow means speed ≤0.02 reduced units/s. "
        "The intervals condition on the saved model fits, split and thresholds.", "",
        "[Population profiles](population_profiles.csv) · [Seed profiles](seed_profiles.csv) · "
        "[Matched trials](paired_airborne_trials.csv) · "
        "[Participant summaries](paired_airborne_participants.csv) · "
        "[Clock and event coverage](alignment.json)", "",
        "Reproduction uses the two source files in this folder. From the project repository, run "
        "r6_population_followup.py with a fresh output directory as its argument. Then run "
        "report_t1_r6_followup.py with the repository and output directory as its two arguments.", "",
    ]
    (DEST / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    summaries = aggregate()
    models = t1_validation()
    paired = paired_slow_motion()
    draw_t1(models)
    draw_r6(summaries, paired)
    report(models, summaries, paired)


