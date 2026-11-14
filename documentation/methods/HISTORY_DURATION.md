# T2: EEG history duration with fixed temporal bins

**T_2. Declared 28 October 2026 before execution.** The user requested
256/512/1,024-ms EEG histories represented by 128 channel-wise mean bins, then
explicitly chose independent within-participant repetitions for P1–P12 and a
paired participant-level summary. This supersedes the registry's earlier T2
deferral; a population timing-gap study is not a prerequisite for this execution.

## Fixed design

- Supplied reduced EEG and wrist X/Y/Z for P1–P12. Train separate models for
  each participant: series 1–6 train, 7 validates, 8–9 test. No pooled population
  training or evaluation on participants excluded from model fitting.
- At 500 Hz, use 128, 256 or 512 consecutive raw EEG samples (nominal histories
  256, 512 or 1,024 ms). Average consecutive non-overlapping groups of 1, 2 or 4
  samples per channel to obtain exactly 128 bins × 32 channels. Bin widths are
  2, 4 and 8 ms. The final raw input is t-1, a fixed 2-ms end gap to target t.
- Retain the original grid `series_start + 128 + k*64`, filtered for eligibility
  at the longest history: t-512 >= series_start and t < series_end. All three
  histories use the same targets in every split, within each participant. The
  256-ms reference is retrained on these common targets; historical scores are
  not its paired baseline. Prediction stride stays 64 raw samples (128 ms).
- Fit per-channel EEG and target StandardScalers on every sample of that
  participant's training series only, as in v2. Transform raw EEG before binning;
  do not fit an additional scaler on validation/test or on the binned matrices.
- Ridge uses the flattened 4,096-value representation. Alpha candidates are
  1/10/100/1,000/10,000, LSQR tolerance .001 and maximum 200 iterations. For each
  history/participant select alpha on validation mean axis MSE, lower-alpha tie.
- CNN + ELU is the existing 3,235-parameter architecture, input (128,32,1),
  Adam/MSE, batch 64, fixed final 15 epochs, seed 42, no early stopping. Reset
  the seed and verify identical initial weights for every history and person.
  Require a GPU computation probe. Ridge runs on CPU. No decoder architecture,
  epoch or preprocessing search is part of this experiment.

## Selection and population summary

1. Complete the fitting/validation phase for all 12 participants and three
   histories before any test evaluation. Save a global frozen selection file
   containing the best history per participant/model (shorter-history tie).
2. Evaluate **all three predeclared histories** on series 8–9. This is a planned
   paired comparison, not a test-based history search. Also report the test
   performance of the validation-selected histories as a secondary result.
3. Primary endpoint is mean X/Y/Z MSE in each participant's own training-standardized
   wrist coordinates. Report per-axis R²/MSE/RMSE/MAE and per-series metrics.
4. Compare 512 versus 256 ms and 1,024 versus 256 ms separately for ridge and
   CNN. Give every participant equal weight: individual paired differences,
   mean/median relative MSE reduction, count improved and participant-bootstrap
   95% intervals (20,000 paired resamples; fixed seed 20261028). Positive relative
   reduction favours the longer history. Also give intervals for absolute MSE
   differences. These four contrasts and intervals are exploratory, without
   multiplicity-adjusted confirmatory claims or an independent-window test.
5. Save input/source/protocol hashes, scalers, common split indices, every ridge
   candidate, CNN weights/history, validation predictions, frozen selections
   and all planned test predictions. Independently replay saved weights and
   rebuild scalers/windows/metrics and participant summaries without training.

## Interpretation limits

The result assesses the consistency of history/averaging effects across these
12 separately trained participant models. It is not zero-shot cross-participant
generalization, leave-one-participant-out evaluation or a pooled population decoder.
One seed leaves optimization variability unresolved. The participants and historical
test series are familiar; the study remains exploratory.

Longer histories also increase averaging and coarsen the effective sampling rate
and the physical duration of each CNN kernel. Equal input dimensions and parameter
counts do not isolate context from smoothing, aliasing or temporal resolution.
Full-resolution and averaging-only controls remain future work; do not claim a
pure history-length effect or an optimal physiological delay from this comparison.
Unknown upstream cleaning prevents causal/real-time preprocessing claims. Reduced
targets and standardized errors are not physical displacement units.
