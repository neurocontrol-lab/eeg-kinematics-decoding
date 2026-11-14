# T1: EEG-to-kinematics timing alignment

Experiment **T_1**, declared 27 October 2026 before its execution. Question: does a window ending earlier improve wrist prediction relative to the current one-sample alignment? This is exploratory P1 analysis: historical test runs 8–9 have already informed the hypotheses. No population, neural-conduction-delay or independent-confirmation claim follows.

## Fixed comparison

- P1, seed 42; whole runs 1–6 train, 7 validates, 8–9 test. Continuous recordings including intertrial intervals, matching v2. Data are the supplied externally reduced EEG and wrist X/Y/Z.
- Inputs: 128 samples × 32 channels (256 ms at 500 Hz); stride 64 samples (128 ms). Window length, channels, models, training budget and run partitions remain fixed.
- End gap means target index minus the final EEG input index. Gaps are **1, 25, 50, 100 samples = 2, 50, 100, 200 ms**. For target `t`, the input spans indices `[t-gap-127, t-gap]` inclusive. Every input precedes its target and remains inside its run. This changes the history interval, not the target timestamp.
- Start with the original v2 target grid `run_start + 128 + k*64`. Retain only targets eligible for all four gaps. All gaps have identical training/validation/test target indices and labels. This drops two leading windows per run relative to v2. The 2-ms model must be retrained on this common set; historical scores are not the paired baseline.
- Fit separate channel and wrist StandardScalers on all samples of training runs only, as in v2. Use float32 inputs/targets. Keep these same transforms for all gaps; validate against the existing six-model scaler metadata.
- **Ridge:** flattened window; alpha candidates 1, 10, 100, 1,000, 10,000; LSQR tolerance 0.001 and maximum 200 iterations, matching the existing reference. Select alpha for each gap using validation MSE averaged over X/Y/Z; report iteration counts and convergence-budget limitations.
- **CNN + ELU:** existing 3,235-parameter regressor (two ELUs), Adam/MSE, batch 64, fixed final 15 epochs. Reset seed 42 and verify identical initial weight hashes for each gap. Train on GPU, require a device computation probe, record versions. Ridge uses CPU. No early stopping, test-selected checkpoints or extra epochs.

## Selection and reporting

1. Fit all declared candidates and compute validation scores before evaluating any test prediction.
2. Independently for ridge and CNN, select the gap with lowest mean validation MSE. Ties select the shorter gap; ridge-alpha ties select lower alpha. Save the frozen selection and a timestamp before test evaluation. Do not refit on validation data.
3. Evaluate only each model's selected gap and its 2-ms common-target baseline on test runs 8–9. If these are the same gap, evaluate once and report that there was no selected alignment change. Do not evaluate the other gaps on test or retune based on test results.
4. Report validation curves; test per-axis MSE/RMSE/MAE/R² and per-run metrics; target-stage and pure-context stage metrics using the previously verified P1 event file. Pure context requires every index from the earliest EEG input through the target to remain in the same phase. These are diagnostic subsets, not separate stage-trained models.
5. Report paired mean prediction-error norms per trial across all 68 held-out trials, with selected-minus-baseline differences. Overlapping windows are not independent replicates. No confirmatory significance threshold is declared.
6. Plot ridge channel-by-lag prediction coefficients, clearly labelled as decoder weights rather than physiological activation/source maps. A full activation-pattern/stability analysis is not part of this pilot.

Exact source/protocol snapshots, input hashes, split indices, model weights, validation predictions, permitted test predictions and numeric reports are retained. Independent verification recomputes indexing/boundaries, train-only scaling, selection, saved metrics, phase assignment and paired-trial summaries. Source data and historical runs are not modified.

## Limits and next decision

Timing affects the effective input history, not necessarily physiology. The current window already contains earlier EEG, so larger gaps may help, hurt or leave performance unchanged. Validation is one run, neural arithmetic can vary despite seeding, and ridge uses the existing bounded solver budget. Supplied reduced preprocessing is not fully documented; causal indexing does not establish causal upstream filtering or real-time suitability. Test-stage R² may be unstable for short stages.

After verification, retain the result whether positive, negative or inconclusive. Review it before finalizing C_7: retrospective stage classification using shrinkage LDA and CNN on the same binned full-stage representation, with exact event labels and a timing-only control. C5 and C3 remain queued. A negative classifier result is not a reason to skip their regression questions.
