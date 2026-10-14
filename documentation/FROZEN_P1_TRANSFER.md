# P1-trained, zero-adaptation cross-participant evaluation

[Completed GPU run](../outputs/frozen-p1-cross-participant-gpu-20260930T125956Z/RESULTS.md)
· [Independent output verification](../outputs/frozen-p1-cross-participant-gpu-20260930T125956Z/verification.json)

The six-model run completed for every target participant P2–P12. Its primary
runs 8–9 scores have negative participant-macro R² on every model/axis pair;
the frozen P1 mean has better macro scores than all learned models. The only
positive primary target scores are P10 Z for CNN + ELU (0.0106) and EEGNet
(0.0121). Source P1 control scores remain positive for all learned models.
All participants' primary runs are weight series with fixed sandpaper. The report includes 45,389
primary target windows and 281,267 windows across all nine target runs.

The [runner](../experiments/evaluate_frozen_p1.py) reuses all six saved models
from the [P1 reference comparison](SIX_MODEL_COMPARISON.md): training mean,
ridge, original CNN, CNN–BiLSTM, CNN + ELU, and EEGNet regression. It never
trains a model or selects parameters using a target participant.

## Frozen protocol

- Source: the six-model P1 GPU run `six-model-comparison-p1-gpu-20260930T095506Z`.
  P1 runs 1–6 fitted weights/scalers; run 7 selected ridge regularization;
  fixed final neural weights were saved after 15/20 epochs.
- Freeze both EEG and target scaler means/scales from P1 training samples.
  Target participants receive those transforms without any fitting.
- Load neural models with `compile=False`, set `trainable=False`, and perform
  GPU inference only. Batch-normalization statistics and every weight must
  remain exactly unchanged. Mean/ridge inference uses CPU.
- Evaluate P2–P12. Primary results use runs 8–9, matching P1 test-run IDs.
  Secondary results cover all nine runs of each target participant. Each
  recording also receives its own metrics and surface-code metadata.
- Preserve 128-sample inputs, stride 64, and a one-sample lead after the last
  input. Inputs and targets must remain inside one recording run.
- Replay P1 test runs 8–9 with the original concatenated batch boundaries.
  Verify targets, sample indices, predictions and metrics against saved P1
  outputs before proceeding to target participants.
  A diagnostic replay found bitwise-identical EEG inputs but GPU LSTM prediction
  variation of up to 0.005582 (mean 0.000056) standardized units across processes.
  The report records per-model maximum/mean differences and enforces prediction
  and aggregate-metric tolerances separately; neural weights remain bitwise fixed.
- Compare every participant's reduced targets with the raw `HS` recordings
  using the existing provenance method. Require matching nine run lengths and
  wrist channels to correlate more strongly than the other same-axis sensors.
- Save per-axis R²/MSE/MAE, all per-run metrics, predictions and provenance
  evidence. Recompute primary metrics from saved arrays. Check source artifact
  hashes and all neural weights after inference.

## Interpretation

Participant-macro statistics give each of the eleven target participants equal
weight; the P1 replay control is excluded. Report both mean and median R² and
the participant distributions. Do not treat overlapping windows as independent
observations for confidence intervals. A negative R² is worse than predicting
that evaluation subset's own target mean. The frozen P1 mean provides an
additional deployable reference without target fitting.

Matching run IDs does not establish matching friction conditions, recording
durations or movement distributions. The report retains those metadata. The
all-run secondary score has a different run composition from the P1 held-out
score and must be labeled accordingly.

The archived runner/results use `friction_levels` for the `CurS` surface codes.
Those are not `BlockType` series labels. The subsequent C1 audit verifies that
runs 8–9 are weight series at fixed sandpaper (`CurS=3` in these local files).
This clarification preserves archived numeric results and source snapshots.

This experiment tests transfer from **one source participant**. It is not
leave-one-participant-out population training, target adaptation, or a physical
coordinate calibration experiment. The existing reduced targets span 0–1 per
participant and the upstream EEG/target processing is undocumented. Frozen
downstream scalers do not undo upstream normalization; error magnitudes refer
to P1-standardized reduced coordinates, not physical distances. The raw folders
verify target provenance but are not substituted for the original training
representation. Evaluating an end-to-end raw-data pipeline requires a separately
specified preprocessing and coordinate definition.

## Run

From the repository in the configured WSL TensorFlow GPU environment:

```bash
~/.venvs/eeg-tf-gpu/bin/python experiments/evaluate_frozen_p1.py
```

The runner requires the saved P1 `.keras` and baseline `.npz` artifacts, reduced
P1–P12 MAT files in `way_eeg_clean/`, and raw `P1/` through `P12/` folders beside
the repository. It creates a separate timestamped result directory. Original
notebooks, recordings, source weights and source results remain unchanged.
