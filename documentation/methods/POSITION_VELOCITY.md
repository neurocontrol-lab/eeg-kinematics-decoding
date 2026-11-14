# T_7 — EEG decoding of wrist position and velocity

Declared 10 November 2026 before T7 fitting. Does EEG improve held-out
estimates of wrist position and signed wrist velocity over simple references?
Compare P1–P12 with independently fitted participant models.

## Matched comparison

- Use verified T_8 inputs, recording boundaries, split indices, EEG
  scaling, and its fixed 2 ms position reference. Train on series 1–6,
  select ridge penalties on series 7, and evaluate series 8–9. The test
  series are weight series at surface code 3. Use the T1 common grid
  (stride 64 samples, 128 ms) and the same unbinned 128 × 32 EEG input
  ending at sample t−1 for target t. The longer T2 inputs are separate.
- Reuse each participant’s T1 2 ms ridge and seed-42 CNN + ELU position
  weights. Check source, weights, raw data and split identity. Use seed 42
  for velocity with identical initial weights and 15 epochs, batch 64,
  Adam and MSE. Match ridge’s five penalties and solver budget (LSQR,
  tolerance .001, 200 iterations). Ridge fits on CPU and CNN on GPU.
- Position is supplied wrist X/Y/Z at t. Velocity is the signed X/Y/Z
  first derivative of the same reduced wrist trace, using a cubic
  Savitzky–Golay 41-sample window at 500 Hz, matching R6’s primary
  smoother. Filter each series independently. Center the derivative on t
  with ±40 ms interior support. At recording edges use the within-series
  polynomial edge evaluation (SciPy mode interp). Keep the exact T1
  timestamps and report how many use edge evaluation.
- Fit a separate velocity scaler to all derived samples of training series
  1–6 only, mirroring the position scaler’s fitting scope. The velocity
  label uses kinematics after t in the interior. It is an offline label;
  the EEG input ends before t. No measured wrist state is a decoder input.
- Use a training-mean reference for each target as the primary denominator.
  Also report zero velocity as a secondary reference. Select ridge’s
  velocity alpha separately per participant on validation XYZ MSE, with
  a lower-alpha tie break. Freeze all choices before new test scoring.

## Metrics and verification

Primary per-participant metric is MSE skill, 1 − model MSE / training-mean
MSE, over the standardized XYZ targets. Positive skill beats the target’s
own training mean. Report velocity minus position skill for each model
on the same held-out samples, mean skill by target/model, axis-wise R²
and MSE, and each test series. Report skill against zero velocity separately.
Use paired participant-bootstrap percentile intervals (10,000 draws,
seed 20261110) and counts of positive participant skills. Each participant
has equal weight. This compares usefulness relative to each target’s
baseline; targets differ in dynamics and noise, so skill is not a direct
measure of what EEG encodes.

Record the exact sources, scalers, alignment, validation choices and local
weight/prediction files. Run a separate test/replay worker after the
global freeze. Reconstruct velocity labels with independent filter
coefficients, replay all velocity validation fits and both targets'
chosen test weights, and compare reused position predictions against
the verified T1 arrays. Keep the source T1 artifacts read-only.

This is an exploratory comparison in familiar participants and test
series. It uses one CNN seed and one derivative definition. Reduced
velocities are in reduced-coordinate units per second. Unknown upstream
preprocessing limits online and physiological interpretations.

