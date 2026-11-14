# R6: movement dynamics and saved-decoder error

**R_6. Declared 28 October 2026 before computing the new dynamics results.**
Question: do observed movement dynamics explain the saved decoders' errors beyond
observed position and experimental event stages? This is retrospective P1
analysis of familiar series 8–9, not a new independent decoder test.

## Fixed data and scope

- Reuse all six saved prediction files from the six-model comparison and the
  verified R1–R5 aligned trace/event file. Retain original target indices,
  128-ms prediction cadence, scalers and decoder weights. No new kinematics
  decoder training, inference or lag/model selection.
- Check the original prediction hashes, reconstruct their reduced-coordinate
  observations/predictions, and match every trace index, phase and trial.
  Keep unassigned/intertrial samples in descriptive scatter plots. Restrict
  primary explanatory fits to event-assigned targets in the 68 trials.
- CNN–BiLSTM is the preselected lead model from the prior analysis. Repeat
  numerical comparisons for the other five references. Metrics of error
  explanation are distinct from wrist-reconstruction R².

## Dynamics and descriptive comparisons

- Differentiate the observed **500-Hz reduced wrist trajectories**, separately
  inside each recording series. Use cubic Savitzky–Golay filters with 41 samples
  (82 ms) as primary and 21/81 samples (42/162 ms) as fixed sensitivities.
  Evaluate first, second and third derivatives at the original decoded target
  indices. Never differentiate interpolated predictions.
- Use a common eligibility mask across all smoothing settings, excluding
  targets within 40 samples of either series edge. Filtering can span trial
  and stage boundaries inside a series. This is retrospective symmetric
  smoothing, not a causal online feature pipeline.
- Record signed XYZ velocity/acceleration/jerk and their norms. Coordinates
  and derivatives have reduced units, not metres or physical directions.
  Per-axis upstream rescaling can change the geometry. Jerk is exploratory
  because a third derivative is particularly sensitive to smoothing.
- Plot observed-versus-predicted XYZ coloured by speed, acceleration and jerk,
  error-versus-dynamics curves, and speed curves within event stages. Report
  bin counts, contributing trials and trial-level variation. Correlations are
  descriptive, without independent-window p-values.
- For a direct scatter audit, define axis anchors using the five most occupied
  0.02-wide observed-position bins in original training series 1–6. Describe
  near-diagonal samples (absolute axis error ≤0.05), and off-diagonal samples
  (absolute axis error ≥0.10) with observed-anchor only, predicted-anchor only,
  both or neither. Report counts/speeds per axis and trial. These geometric
  regions describe the streaks; their relation to error is not independent
  evidence that a predictive model improves. Thresholds are fixed, not tuned.

## Explanatory error models

Predict the original decoder's reduced-coordinate **XYZ error norm**, without
changing its kinematics predictions. Fit on series 8 and evaluate on 9, then
reverse. Every evaluated trial is absent from its explanatory fit. These two
folds explore transfer of an error relationship between familiar test series;
they do not convert the decoder's historical test data into new confirmation.

- Feature groups: **P** observed XYZ position with all degree-2 terms;
  **S** eight fixed event-stage indicators; **D** signed XYZ velocity and
  acceleration plus speed/acceleration norms, with all degree-2 terms.
  Jerk group **J** uses signed XYZ jerk and its norm with degree-2 terms.
- Compare a fitted constant, P, S, D, P+S, P+D, S+D, P+S+D, and the diagnostic
  P+S+D+J. Construct polynomial terms within each continuous group; no
  interactions across groups. Run the same nine models at all three windows.
- Standardize features using each fitting series only, with trial-balanced
  weights. Ridge alpha is fixed at 10 with an intercept and no hyperparameter
  search. Fit weights have mean 1 and equal total weight per trial. Do not
  clip negative error predictions; record their frequency.
- Primary score: mean of the 68 per-trial explanatory squared errors. Also
  report each series, trial-balanced MAE and error-target R². The fitted
  constant is a cross-series prediction reference; conventional error-target
  R² uses the evaluated targets' weighted mean in its denominator.
- **Primary contrast:** CNN–BiLSTM at 41 samples, P+S+D versus P+S. Report
  paired per-trial loss differences and the fraction of trials improved.
  Inspect added-stage benefit (P+D versus P+S+D) separately. A lower error
  after adding dynamics is association, not proof of neural mechanisms.
- Exploratory uncertainty for the primary contrast: circular blocks of four
  consecutive trials, resampled independently within each series, 5,000
  replicates, seed 42. Repeat blocks 2/8 with 2,000 replicates. These are
  pointwise descriptive intervals, conditional on the fitted explainers;
  they do not refit models or cover population/seed uncertainty.

## Evidence and next decision

Save exact executed sources and this protocol, hashes, numerical summaries,
per-trial/stage/region tables, out-of-series explanatory predictions and fit
parameters locally, plots and a short report. An independent verifier must
recompute source matching, stage/trial eligibility, derivatives, fit scaling,
saved coefficient replays, metrics and the primary interval. Verify source
files remain unchanged.

Interpret smoothing stability, sample coverage and series consistency before
deciding whether event stages, movement dynamics or both merit follow-up.
Stage-specific decoder training, EEG stage classification, automatic
segmentation and real-time error prediction remain separate experiments.
