# Pipeline changelog

## 2026-10-20 — Workspace organization and coordinate-axis audit

- Group experiments into stable numbered families, including original/corrected baseline notebooks, with commands and status in family READMEs.
- Centralize raw/reduced-data paths and move local participant data, downloaded reductions, reference papers and presentation artifacts to their designated locations.
- Organize dataset/method/reproducibility documentation and update navigation and execution references.
- Preserve saved predictions, models, metrics, verification records and executed source/protocol snapshots. Resolve historical data identities through the shared path helper.
- Publish the 3,528-trial axis audit: position Z is vertical (+ upward), force X is load/lift and force Z is grip. X/Y anatomical directions remain unresolved.

This is the central record of pipeline versions. Version numbers identify pipeline revisions; they do not imply an improvement in model architecture or performance.

## Saved-model inference benchmark (2026-10-14)

[Results](outputs/inference-benchmark-p1-20261014/RESULTS.md)
· [Runner](experiments/06_inference_benchmark/benchmark_inference.py)

- Measure warmed, synchronized CPU/GPU inference for six frozen P1 references
  at batch sizes 1 and 64, with 300 calls per configuration and identical inputs.
- Preserve median/p95 timings, round medians, raw timing arrays and replay/source
  checks. Verify all 20 timing arrays against their reported median/p95 values.
- Single-window CPU calls are faster for CNN variants; GPU benefits batch
  throughput and gives CNN–BiLSTM a modest single-window speedup. This is model
  latency, not end-to-end real-time validation.

## Category 1: condition audit and P1 classification (2026-10-14)

[Methods](documentation/methods/CONDITION_CLASSIFICATION.md)
· [C1 audit](outputs/condition-audit-20261014T115618Z/RESULTS.md)
· [C2 results](outputs/weight-classification-p1-gpu-20261014T115817Z/RESULTS.md)

- Audit named metadata columns for 3,528 supplied P1–P12 trials, including
  current/previous weight/surface and split coverage. No held-out current joint
  condition is absent from v2 training; series-type composition differs.
- Fit trial-grouped P1 weight classifiers at fixed sandpaper. Runs 1/4 train,
  7 validates, 8/9 test; full LEDOn-to-hand-stop windows stay within trials.
  Verify all 170 trial clocks and fit scalers only on training-trial samples.
- Compare validation-selected logistic regression and a fixed-epoch GPU CNN +
  ELU classification head with a training-majority reference. Held-out trial
  balanced accuracy is 32.48%, 27.73% and 33.33%, respectively.
- Enumerate 1,156 within-run circular trial-label offsets per learned model.
  Upper-tail fractions 0.592/0.924 do not support reliable discrimination.
- Verify saved predictions, 21 metric subsets, 2,312 reference offsets,
  training-only scaler statistics, labels, boundaries and source snapshots.
  Keep the negative result; no test-driven tuning or C3–C6 training occurs.

## Frozen P1 cross-participant evaluation (2026-09-30)

[Methods](documentation/methods/FROZEN_P1_TRANSFER.md) · [Runner](experiments/03_cross_participant_decoding/evaluate_frozen_p1.py) · [Completed run](outputs/frozen-p1-cross-participant-gpu-20260930T125956Z/RESULTS.md)

- Reused all six saved P1 models and P1 training scalers without target fitting,
  fine-tuning or batch-normalization updates. Neural inference ran on GPU.
- Evaluated P2–P12 on weight runs 8–9 (45,389 windows, fixed sandpaper), with all
  nine target runs as a secondary evaluation (281,267 windows). Reported each
  participant, run and wrist axis; participant-macro averages exclude P1.
- Every model/axis primary macro R² is negative. The frozen P1 mean has better
  macro scores than the learned models. Only P10 Z is slightly positive for
  ELU/EEGNet; strong within-P1 results do not transfer reliably in this protocol.
- Extended raw/reduced wrist provenance checks to all twelve participants.
  All nine run lengths match; minimum wrist Pearson r is 0.9999231.
- Verified P1 replay, unchanged weights/source hashes, and saved metrics/window
  boundaries. Independent verification covers 72 prediction files and 750 metric
  subsets. GPU LSTM replay variation is explicitly recorded.
- Existing upstream processing and physical coordinate units remain unresolved;
  this is transfer from one source, not population-model evaluation.

## Six-model P1 reference comparison (2026-09-30)

[Methods](documentation/methods/SIX_MODEL_COMPARISON.md) · [Runner](experiments/02_model_comparison/compare_six_models.py) · [Completed run](outputs/six-model-comparison-p1-gpu-20260930T095506Z/RESULTS.md)

- Added training-mean and ridge baselines, an ELU-only CNN ablation, and an
  EEGNet regression reference alongside the two unchanged v2 architectures.
- Reused v2 P1 train/validation/test runs and training-only scaling. Neural
  models ran on the RTX 3060 GPU with seed 42 reset per model; mean/ridge used CPU.
  Ridge alpha 100 was selected using validation MSE, without test tuning.
- ELU-only CNN R² improved from 0.547/0.399/0.451 to 0.746/0.713/0.678.
  EEGNet regression scored 0.758/0.683/0.663; retrained CNN–BiLSTM scored
  0.809/0.783/0.748. Ridge scored 0.558/0.420/0.462.
- Verified saved prediction metrics and reloaded-model inference within recorded
  GPU numerical tolerances. The initial source review corrected the spatial
  max-norm axis for time-first EEGNet kernels before that model trained.
- Single P1 seed and friction-3 test runs limit interpretation. No cross-participant
  claims or significance claims are made. Original notebooks/results remain unchanged.

## Repository organization (2026-09-30)

- Shortened the notebook filenames to experiments/01_continuous_kinematic_decoding/original_baseline_v1.ipynb and experiments/01_continuous_kinematic_decoding/corrected_baseline_v2.ipynb and updated their references, including v1's source-notebook snapshot path.
- Separated the README's pipeline descriptions from its file and code structure.
- This update changes naming and documentation only; training protocols and saved results are unchanged.

## v2 — Corrected baseline (2026-09-29)

[Notebook](experiments/01_continuous_kinematic_decoding/corrected_baseline_v2.ipynb) · [Detailed methods and limitations](documentation/methods/PIPELINE_V2_CORRECTED_BASELINE.md) · [P1 GPU run](outputs/pipeline-v2-corrected-baseline-p1-gpu-20260929T173627Z/RESULTS.md)

- Added `PARTICIPANT_ID` at the top of the notebook. It selects the matching EEG, kinematics, run-boundary, and lift-metadata files and labels each new result. The reduced P1–P12 files align; the extra P55 EEG and kinematics files have different sample counts and fail the alignment check.
- Split complete recording runs before scaling or windowing. For each supplied nine-run participant, runs 1–6 train, run 7 validates, and runs 8–9 test.
- Fit separate EEG-channel and wrist-target scalers on training samples only. Keep each 128-sample, stride-64 window and its following target inside one run.
- Use validation data during training and evaluate test runs only after both fixed training schedules finish. Set seed 42 and report aggregate and per-test-run scores.
- Identified the three reduced kinematics columns as wrist X/Y/Z using the [source-data comparison](documentation/dataset/KINEMATICS_PROVENANCE.md).

The CNN and CNN–BiLSTM definitions and their 15/20-epoch schedules are unchanged from v1. The saved P1 GPU run reports CNN–BiLSTM R² of 0.784 / 0.757 / 0.726 for wrist X/Y/Z. P2 has passed loading, preprocessing, windowing, and model-build checks but has **not** been trained or evaluated. The P1 held-out runs contain friction condition 3 only; no cross-subject or population-model result has been produced.

## v1 — Original fork baseline

[Notebook](experiments/01_continuous_kinematic_decoding/original_baseline_v1.ipynb) · [Methods and limitations](documentation/methods/ORIGINAL_BASELINE.md) · [P1 CPU run](outputs/original-p1-cpu/RESULTS.md) · [P1 GPU run](outputs/original-p1-gpu/RESULTS.md)

The original notebook uses overlapping windows with a random window-level split, scales data before the split, and passes the test partition to training as validation data. Its saved P1 runs reproduce that protocol. The v1 and v2 scores use different evaluation settings and target scaling, so they should not be interpreted as a direct model-performance comparison.
