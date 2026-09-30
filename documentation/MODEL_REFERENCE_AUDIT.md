# Existing model reference audit

Reviewed 30 September 2026. Scope: notebook definitions, saved P1 v2 model
configurations, execution snapshot, numeric results, and EEGNet's author implementation.
No model or training protocol was changed, and no new training was performed.

Subsequent experiment: the [six-model P1 comparison](SIX_MODEL_COMPARISON.md)
has now [completed](../outputs/six-model-comparison-p1-gpu-20260930T095506Z/RESULTS.md).
It adds the basic baselines and activation/EEGNet variants, checks saved metrics,
and reloads neural models for inference. Multi-seed, phase-specific and additional
participant validation remain outstanding. The initial audit below is preserved.

## Decision

Keep both existing architectures as frozen project references. The CNN is an
EEGNet-like affine regressor; the CNN–BiLSTM is a nonlinear sequence regressor.
Their P1 v2 results provide an initial within-participant comparison. Neither
is yet a validated population baseline or evidence of neural causation.

## Relationship to EEGNet

Sources: [author implementation](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py)
and [EEGNet paper](https://arxiv.org/abs/1611.08024v4).

| Feature | Project CNN | Author EEGNet implementation |
| --- | --- | --- |
| Temporal / spatial / separable filters | 8 / multiplier 2 / 16 | Same default counts |
| Input convention | Time × electrodes | Electrodes × time; transpose kernels accordingly |
| Nonlinearities | None | ELU after spatial and separable blocks |
| Temporal pooling | Factor 4 after separable convolution | Factor 4 before separable convolution, then factor 8 |
| Weight constraints | None | Spatial and output max-norm constraints |
| Output | Three linear coordinates, MSE | Class probabilities |

The shared structure supports calling this CNN EEGNet-like; it does not prove
direct derivation. A linear output is suitable for regression. Removing hidden
nonlinearities and changing pooling substantially changes the feature extractor.
With fixed batch-normalization statistics and disabled dropout, this project's
CNN is affine in the input. Its training dynamics differ from ordinary linear
regression. Preserve it under an explicit name rather than silently modifying it.

Temporal kernel lengths also require sampling-rate context. The local files lack
sampling-rate metadata; 500 Hz is supported by the original dataset documentation.
Do not interpret copied sample counts as matching the temporal scales of EEGNet's
default configuration.

## Local implementation evidence

- Both saved `.keras` configurations confirm the notebook layer sequences:
  CNN has linear convolution/output activations; CNN–BiLSTM has a ReLU temporal
  convolution, bidirectional LSTM, ReLU hidden dense layer, and linear output.
- The execution snapshot hardcodes dimensions 128 and 32; the current notebook
  uses WINDOW and N_CHANNELS. For P1 these resolve to the same architecture.
- CNN–BiLSTM's two directions read only the observed window. This is valid for
  prediction after the complete window; it does not establish streaming latency
  or memory carried between windows.
- Total parameter counts are 3,235 and 61,347. These are different-capacity
  references, so a performance gap alone does not isolate the value of recurrence.
- V2 code separates runs before scaling/windowing and evaluates test predictions
  after both fixed training schedules. This audit inspected that code; it did not
  independently re-execute data preparation or training.

## Existing performance evidence

Source: [saved P1 v2 run](../outputs/pipeline-v2-corrected-baseline-p1-gpu-20260929T173627Z/RESULTS.md).

| Model | Test R² X / Y / Z | Final train MSE | Final validation MSE |
| --- | --- | ---: | ---: |
| CNN | 0.547 / 0.399 / 0.451 | 0.322 | 1.931 |
| CNN–BiLSTM | 0.784 / 0.757 / 0.726 | 0.129 | 0.310 |

Both test runs have positive R² on all axes. The large CNN validation gap needs
investigation: possible explanations include run distribution differences,
outliers, overfitting, and batch-normalization statistics. The existing numbers
do not distinguish those explanations. CNN–BiLSTM's epoch-16 loss spike also
merits checking reproducibility; it is not by itself evidence of a code defect.

## Validation needed before further pipeline changes

1. Recompute metrics from saved prediction arrays and confirm model reload
   reproduces predictions for the saved preprocessing and target indices.
2. Repeat the unchanged P1 protocol across predeclared seeds to quantify training
   variability. Inspect train and validation predictions in inference mode.
3. Compare a training-target-mean predictor and regularized linear regression
   using the same EEG windows and splits. Fit/select them without test tuning.
4. Evaluate movement and rest separately using verified event/sample alignment;
   inspect chronological predictions and validation-run target distributions.
5. Run unchanged models on additional participants and predeclared run splits
   covering friction conditions. Keep participant/run reporting separate.
6. Once these checks are complete, add a separately named EEGNet regression
   reference with documented adaptation. Retain the existing CNN and its results.

Historical v1 results remain reproduction references. Use v2 for prospective
comparisons. Agreement with EEGNet's structure alone cannot validate the cleaned
EEG, remove artifact confounds, or establish cross-participant generalization.
