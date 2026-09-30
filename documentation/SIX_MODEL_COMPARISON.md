# Six-model P1 comparison

This experiment extends the frozen v2 reference with two basic predictors, an
ELU-only CNN ablation, and an EEGNet regression architecture reference. It uses
P1 only; participant selection and the original notebooks remain unchanged.

## Shared protocol

The [runner](../experiments/compare_six_models.py) executes the preparation cells
of `pipeline_v2.ipynb` directly. It checks the expected P1 partitions: runs 1–6
train (18,516 windows), run 7 validates (1,867), and runs 8–9 test (3,642).
All models receive the same training-only standardized EEG and wrist targets,
128-sample windows, stride 64, and target one sample after the last EEG input.
Windows do not cross run boundaries. All training and validation-based selection
finish before test scores are computed.

| Model | Definition | Training |
| --- | --- | --- |
| Training mean | Constant training-window target mean per axis | CPU; no tuning |
| Ridge | Flattened 128 × 32 EEG window, intercept, L2 penalty | CPU; alpha selected using validation MSE |
| CNN | Existing v2 definition | GPU, 15 epochs |
| CNN–BiLSTM | Existing v2 definition | GPU, 20 epochs |
| CNN + ELU | Existing CNN plus ELU after spatial and separable batch normalization | GPU, 15 epochs |
| EEGNet regression | EEGNet feature extractor with linear three-coordinate head | GPU, 15 epochs |

All neural fits use Adam, MSE, batch size 64 and seed 42, reset before each model.
Final-epoch weights are evaluated; there is no test-based epoch selection. The
ELU-only model has the same 3,235 parameters as the CNN, and the runner checks
that both start with identical weight arrays. Adding ELU is its only layer change.
The fixed 15-epoch CNN schedule provides a controlled first architecture comparison;
it does not establish optimally tuned performance for each architecture.

Ridge uses scikit-learn's LSQR solver (tolerance 0.001, maximum 200 iterations).
Candidates are alpha 1, 10, 100, 1,000, and 10,000. The lowest validation MSE
selects the trained candidate; validation samples are not added to its training.
The runner records each candidate's iterations and score. These settings define
a practical regularized linear reference, not an exhaustive hyperparameter search.

## EEGNet adaptation

Reference: [authors' EEGNet implementation](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py).
The input and kernels are transposed to the project's time-first convention.
The regression variant uses temporal length 64 with 8 filters, spatial multiplier
2, 16 separable filters of length 16, two ELUs, temporal pooling factors 4 and 8,
dropout 0.5, spatial max-norm 1 and output max-norm 0.25. The class head is replaced
with three linear coordinates, trained with MSE. It has 1,891 total parameters.
Spatial max-norm uses axis 1 in this time-first implementation, corresponding
to axis 0 (electrodes) in the authors' channel-first spatial kernel.

This reproduces the specified feature-extractor structure, not the paper's
classification experiments or an established wrist-decoding model. Temporal
kernels retain their sample lengths; no sampling-rate adjustment is made.
Changing pooling, dropout, constraints and nonlinearities together means its
comparison cannot isolate the effect of activations; the ELU-only ablation can.

## Outputs and checks

New results are saved under `outputs/six-model-comparison-p1-gpu-<UTC timestamp>/`.
The runner requires an actual GPU and confirms a matrix multiplication is placed
on `/GPU:0`. It stores protocol, source hashes, Git state, environment versions,
validation scores, learning histories, per-axis and per-test-run metrics, plots,
and source snapshots. GPU training remains numerically nondeterministic across
systems despite seeding.

Neural test inference reloads the saved models; the first 64 outputs are checked
against direct inference. Metrics are recomputed from saved prediction arrays.
Model weights and `.npz` arrays remain local under existing ignore rules.

## Interpretation limits

Only one seed and participant are tested. Both test runs and the validation run
contain friction condition 3. Overlapping test windows must not be treated as
independent replicates for significance. Positive R² does not establish neural
causation; artifact controls and movement/rest analysis remain outstanding.
The training-mean baseline can have negative R² because R² uses the test mean
as its reference. MSE and MAE are in training-standardized target units.

Existing models are retrained with a seed reset before each fit, rather than
reusing historical scores. This changes random initialization relative to the
old notebook's construction order, so the newly run CNN–BiLSTM need not reproduce
its previous score. Use this common-protocol comparison for model differences.
