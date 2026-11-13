# EXP-042 / T1: population EEG-to-kinematics timing alignment

Declared 28 October 2026 before computing the new validation or test results.
User-authorized extension of EXP-012. Question: does validation-selected EEG
history alignment improve within-participant wrist decoding across P1–P12?
This trains separate participant decoders, not a population-trained or
leave-one-participant-out model. Historical P1 tests and prior cross-participant
reports have informed planning; all results remain exploratory.

## Fixed comparison

- Include P1–P12, excluding mismatched P55. Require finite, aligned EEG (32
  channels), wrist XYZ, and nine ordered recording-series boundaries at 500 Hz.
  Reuse the condition audit and validate its metadata source hashes. Series
  1–6 train, 7 validates, 8–9 test. The latter are weight series at surface
  code 3 for all participants. Training also contains friction/mixed series.
- Keep 128 input samples (256 ms), stride 64 (128 ms), and the existing
  series-local target grid starting at series_start + 128.
- End gaps: 1/25/50/100/250/500 samples = 2/50/100/200/500/1000 ms. For target
  t, input is [t-gap-127, t-gap], inclusive. Retain only targets eligible for
  every gap, with no crossing recording boundaries. This removes eight
  leading grid points per series relative to v2. All gap comparisons use
  identical targets; retrain the 2-ms reference on this common grid.
- Fit EEG-channel and wrist-axis StandardScalers separately for each
  participant, using all samples of training series only. Reuse each
  participant's transforms across all gaps and seeds. No target-participant
  test calibration or pooled participant normalization.
- Ridge: flattened windows; alphas 1/10/100/1000/10000, LSQR tolerance .001,
  maximum 200 iterations. Select alpha separately at each gap by validation
  XYZ mean MSE, breaking ties toward lower alpha. Record solver iterations
  and budget saturation. CPU fits, retaining the pilot's bounded solver.
- CNN + ELU: same 3235-parameter model as EXP-012, Adam/MSE, batch64,
  final15 epochs, no early stopping. Seeds42/43/44. Match initial weights
  across gaps within each seed and record hashes. Require a working GPU
  probe and record runtime versions. No extra training or test-selected
  checkpoint. These are repeated fits, not an ensemble.

## Selection and test isolation

For each participant, ridge chooses one gap by minimum validation MSE.
CNN chooses one gap by minimum mean validation MSE across the three seeds.
Ties select the shorter gap. Complete all participants' validation fits and
freeze every gap/alpha choice in selection.json before producing any new
test predictions. Test only the selected gap and the 2-ms reference for each
model/seed. If selected gap is2ms, evaluate once. Do not inspect test scores
of other gaps or use test results to change any choice.

Checkpoint every completed fit; resume only with unchanged source/protocol
hashes and verified weight/prediction files. Preserve original EXP-012 outputs.
Do not reuse old P1 fits because its common targets and gap grid differ.

## Population reporting

Primary metric per participant/model is mean standardized XYZ test MSE.
Average the CNN metrics across seeds first, then give all12 participants equal
weight. Primary contrast is selected-minus-2ms participant MSE. Positive
means worse. Also report participant percentage MSE reduction
100*(baseline-selected)/baseline, improved/unchanged/worse participant counts,
and axis-wise MSE/RMSE/MAE/R2 and each test series separately. CNN mean scores
refer to repeated fits, not predictions averaged into an ensemble.

Use participants as the resampling unit. Report exploratory95% percentile
bootstrap intervals for the mean paired MSE difference and mean participant
percentage reduction (10000 participant resamples, seed20261028). Average
seeds before resampling; windows/seeds are not independent participants.
These intervals describe this dataset sample, not a representative population
of all people. A selected2ms gap contributes a paired difference of zero.

Plots: participant-by-gap validation heatmaps and normalized validation curves;
paired baseline/selected test scores; participant percentage changes with seed
variation; selected-gap frequencies; per-axis participant results. Show all12
participants, scales/units, metric direction, and variability. Validation curves
may cover all gaps; test plots cover permitted selected/reference gaps only.
Keep numeric CSV/JSON and the exact executed sources/protocol alongside plots.

## Verification and boundaries

A separate no-fitting verifier recomputes source/input hashes, series/condition
coverage, train-only scalers, common indices, candidate/seed validation metrics,
alpha/gap selections, saved-weight validation/test prediction replays, permitted
test configurations, participant/seed aggregation and bootstrap intervals.
Record exact checked counts and solver-budget limitations.

No stage-specific analysis is required for this extension. Event-aligned
population R6 and phase reports are separate work. No physiological-delay,
causal upstream preprocessing, unseen-participant generalization or full
condition-generalization claim follows. Supplied reduced data have partially
undocumented preprocessing. Longer EEG histories and temporal mean bins are
EXP-018 / T2, deferred until after this fixed-duration T1 study.
