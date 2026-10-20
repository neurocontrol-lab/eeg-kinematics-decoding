# Additional results from existing decoders: first P1 analysis

[Results and plots](../../outputs/saved-decoder-analysis-p1-20261013T172654Z/RESULTS.md)
· [Verification](../../outputs/saved-decoder-analysis-p1-20261013T172654Z/verification.json)
· [Analysis runner](../../experiments/04_stage_decoding/saved_model_analysis/analyze_saved_decoders.py)

This completes an initial R1–R5 analysis for P1 held-out runs 8–9. It uses the
six-model reference experiment only. No
training, inference regeneration, model selection or adaptation occurs.

## Chronological reconstruction and event alignment

Saved target indices recover each decoded sample's position in the continuous
recording. The source cadence is 64/500 = 128 ms. The raw EEG and kinematics
sampling rates are explicitly 500 Hz, and the reduced and raw run lengths match.
The decoded targets also reproduce the reduced kinematics after undoing the
source training scaler. Plot lines connect existing samples; they do not add
new predictions between them.

The AllLifts/WS clock differs from the continuous HS clock by **2 seconds** in
both evaluated runs. The analysis locates exact EEG sample matches for three
trial windows per run to infer that offset, then checks the first 100 EEG and
wrist samples of **every one of the 68 trial windows**. Event times require the
verified offset before assignment to HS/reduced sample indices. Simply adding
`StartTime` and relative event times without this conversion would mislabel
phases in these local files.

Event intervals are preparation, reach, preload, loading, airborne before the
lowering cue, after the lowering cue, release, and return. The airborne interval
contains lifting and holding; the lowering-cue interval can contain reaction
delay. No specific event is invented to separate those sub-stages. Intertrial or
unassigned targets remain a distinct category and are not automatically labeled
stationary.

## What the first results show

For the six-model CNN–BiLSTM, mean reduced-coordinate Euclidean error is 0.3014
during reach and 0.2525 during return, compared with 0.0589 during preparation
and 0.1016 in the airborne interval. Trial-level summaries also separate those
intervals: median trial-mean error is 0.2573 for reach, 0.2478 for return, 0.0504
for preparation, and 0.0798 airborne. These describe pooled-model performance;
they do not yet establish that separate stage decoders are necessary.

Phase assignment initially follows the target sample. A second subset requires
all 128 input samples and the target to share that phase. There are **no pure
preload windows**, only 26 pure loading windows, and four pure release windows.
This limits a fair stage-specific comparison at the existing window length.

At the exploratory low-speed threshold of 0.02 reduced units/s with 42-ms
smoothing, 23.8% of targets are low speed. Their mean error is 0.0778 versus
0.1464 for moving targets. Sensitivity results vary the threshold and smoothing
duration; this definition is not a validated physical rest detector.

Dense-region analysis uses both training-defined observed-position bins and a
separate descriptive 2D observed/predicted scatter grid. It reports occupancy,
phase composition, speed composition, and errors inside/outside dense cells.
Density alone must not be equated with rest.

Bias summaries and a descriptive time-shift correlation scan supplement the
error curves. The latter is an inspection of held-out predictions, not a
validation-selected change to the model's prediction lag.

## Files and reproduction

- `RESULTS.md`: findings, plots, motion-threshold sensitivity and density tables.
- `results.json`: all six models' aggregate, phase, pure-window, motion, density
  and descriptive lag summaries.
- `phase_metrics.csv`: per-axis error and bias by phase/subset.
- `trial_metrics.csv`: error by trial; `trial_phase_errors.csv` and
  `trial_phase_summary.json`: trial-level phase errors and medians/interquartile ranges.
- `trial_events.json`: verified event boundaries and trial-condition metadata.
- `aligned_predictions.csv`: local per-sample trace data, ignored by Git.
- `verification.json`: 21,852 trace rows, 60 aggregate/phase checks, pure-window
  counts, all 68 trial-clock checks, unchanged source hashes and source snapshot.

From the repository in the existing WSL scientific Python environment:

```bash
~/.venvs/eeg-tf-gpu/bin/python experiments/04_stage_decoding/saved_model_analysis/analyze_saved_decoders.py
```

The analysis is CPU-only and does not require TensorFlow inference. It reads
P1 raw HS/WS and AllLifts files in the configured raw-data directory and the saved prediction
artifacts locally. A timestamped output directory is created. `--output-dir`
recomputes in an explicitly selected analysis directory under `outputs/`.

To verify a generated directory and produce the independent per-trial phase
summaries:

```bash
~/.venvs/eeg-tf-gpu/bin/python experiments/04_stage_decoding/saved_model_analysis/verify_saved_decoder_analysis.py outputs/saved-decoder-analysis-p1-20261013T172654Z
```

This first pass does not cover frozen-transfer P2–P12 phase analysis, trial-stage
classifier training, or the C5 cross-stage decoder matrix. Those remain separate
extensions. Error units are reduced coordinates, not physical distances;
overlapping windows are correlated, and short-stage R² can be unstable.

## Added trial-average plots

[Per-run averages and elevated-error intervals](../../outputs/saved-decoder-analysis-p1-20261013T172654Z/trial_averages/RESULTS.md)
supplement the retained run-wide and individual-trial plots. Each run averages
34 LEDOn-aligned trials in nearest-centred 128-ms bins, without interpolation
or new predictions. Coordinate shading shows trial interquartile ranges.
The error norm is computed for each original prediction before averaging.

High-error intervals compare each trial's error with its own preparation-period
mean error. A circular moving-block bootstrap over consecutive trials uses
5,000 resamples, seed 42, and blocks of four trials. A centered standardized
maximum statistic supplies simultaneous 95% bands across eligible time bins,
six model outputs and both runs. At least 28/34 trials must contribute to a
tested bin. Sensitivity uses blocks of two/eight trials and 2,000 resamples each.

These are exploratory, baseline-relative bands, not a preregistered confirmatory
test or a population-level claim. Trials differ in weight and stage timing;
LEDOn alignment can blur later stages. Block resampling approximates local serial
dependence but assumes approximate stationarity. Sparse preparation estimates,
overlapping input history and reduced-coordinate units still limit interpretation.

```bash
~/.venvs/eeg-tf-gpu/bin/python experiments/04_stage_decoding/saved_model_analysis/average_saved_trials.py outputs/saved-decoder-analysis-p1-20261013T172654Z
```

The runner adds `trial_averages/` without replacing the original plots or source
predictions. Orange highlighting is determined by the simultaneous paired-excess
band; pointwise bands on the mean-error panel only describe mean uncertainty.

`experiments/04_stage_decoding/saved_model_analysis/verify_trial_averages.py <analysis-directory>` checks the 534 plotted
curve rows against saved trial arrays, source hashes, trial pairing, coverage,
positive lower-bound flags and interval endpoints. All 12 average plots and the
original run/trial plots are present. [Verification record](../../outputs/saved-decoder-analysis-p1-20261013T172654Z/trial_averages/verification.json).

Dataset event definitions: [authors' utility](https://github.com/luciw/way-eeg-gal-utilities/blob/master/Utilities/WEEG_GetEventsInHS.m).
The clock conversion above is additionally verified against the local recordings.
