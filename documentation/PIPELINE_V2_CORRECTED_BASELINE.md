# Pipeline v2: Corrected Baseline

The [v2 notebook](../pipeline_v2_corrected_baseline.ipynb) is the first evaluation-corrected
pipeline for decoding one participant's EEG into wrist position. Set
`PARTICIPANT_ID` in the first code cell to select P1–P12. It is separate from the
[original notebook baseline](ORIGINAL_BASELINE.md), whose notebook, models, and
CPU/GPU results remain unchanged. The root [changelog](../CHANGELOG.md) records the version history. Its first [GPU run, on P1](../outputs/pipeline-v2-corrected-baseline-p1-gpu-20260929T173627Z/RESULTS.md)
contains the full metrics and plots.

## What changed, and how

| Original pipeline issue | Change in this notebook |
|---|---|
| A random split assigned overlapping windows from the same recording to train and test. | Read the selected participant's one-based run starts from `session_start_P*.mat`, convert them to zero-based boundaries, then assign whole runs. For each supplied nine-run participant: **1–6 train, 7 validation, 8–9 test**. No run is shared between splits. |
| EEG and kinematic values were scaled before splitting, so test statistics affected training inputs. | Fit a separate `StandardScaler` for the **32 EEG channels** and **three wrist axes**, calling `partial_fit` only on samples from that participant's training runs. Apply those fitted transforms to all splits. MSE and MAE now use standardized target units. |
| A window could cross a concatenated run boundary. | Generate windows inside each run's `[start, end)` interval. Each input contains 128 EEG samples, starts advance by 64, and its target is the immediately following sample. Assert that each target index remains inside its run. |
| The test set was passed to `fit` as validation data. | Pass only the validation run to `validation_data`. Evaluate the held-out test runs after both fixed training schedules finish. The test set is not used for early stopping or epoch selection. |
| Initialization and training were unseeded. | Set Python, NumPy, and TensorFlow/Keras seed 42. GPU execution can still differ numerically across systems. |
| The three reduced `kin` columns were described as unnamed glove or finger outputs. | Label them **wrist X, Y, Z** using the [sample-aligned provenance check](KINEMATICS_PROVENANCE.md) against the original recordings. |
| The new experiment was tied to P1 by filenames and report labels. | Derive all four input filenames, the run split, plot labels, `results.json` subject, and output directory from `PARTICIPANT_ID`. Check EEG/kinematic sample alignment before scaling or training; reject a mismatched participant with a clear error. |

The CNN and CNN–BiLSTM architectures stay as in the original experiment:
3,235 and 61,347 parameters, Adam with MSE loss and MAE reporting, batch size
64, and 15 and 20 epochs respectively. The CNN still has no nonlinear
activation. This first experiment measures the corrected data and evaluation
protocol before testing architecture changes. Its fixed final-epoch weights
are evaluated; there is no checkpoint selection.

## First GPU run: P1

| Split | Runs | Windows | Lift trials in metadata |
|---|---|---:|---:|
| Train | 1–6 | 18,516 | 192 |
| Validation | 7 | 1,867 | 34 |
| Test | 8–9 | 3,642 | 68 |

| Model | Wrist X R² | Wrist Y R² | Wrist Z R² |
|---|---:|---:|---:|
| CNN | 0.547 | 0.399 | 0.451 |
| CNN–BiLSTM | 0.784 | 0.757 | 0.726 |

The [run report](../outputs/pipeline-v2-corrected-baseline-p1-gpu-20260929T173627Z/RESULTS.md)
also separates test runs 8 and 9 and shows learning curves and predictions.
The CNN's final validation MSE on run 7 was 1.931 versus training MSE 0.322;
one validation run is an unstable basis for model selection. These scores and
the original random-window scores assess different splits and use different
target scaling, so differences cannot be attributed to a model improvement.

The participant selector was checked with P2 through loading, run splitting,
training-only scaler fitting, window construction, and both model builds. It
produced 22,033 training, 2,356 validation, and 4,642 test windows; model sizes
remained 3,235 and 61,347 parameters. P2 was **not trained or evaluated** in
this check. The P1 scores above are the previously saved corrected run; renaming its files
and protocol label did not recompute its metrics.

## What this does not resolve

- Only participant P1 has been trained and evaluated so far; selecting another
  participant in the notebook does not itself establish cross-subject performance.
  There is no population model,
  subject fine-tuning, or cross-lab test yet.
- For P1, all three held-out runs (7–9) contain **friction condition 3 only**. The first
  run-level test therefore does not measure performance across all friction
  conditions. The notebook reports friction levels for each selected participant's split.
  A later run-based evaluation should include the mixed-friction
  runs without using the final test runs for tuning.
- The validation set contains just one run. Repeating the holdout across runs
  will provide a more stable view of run-to-run variation.
- The cleaned target columns are identified as wrist coordinates, but their
  exact upstream filtering, scaling recipe, and physical units remain unknown.
  EEG cleaning and model architecture have not been revisited.

To reproduce or extend this experiment, open the notebook from the repository
directory with `way_eeg_clean/` present, set `PARTICIPANT_ID` in the first code
cell, select the WSL **Python (EEG TensorFlow GPU)** kernel or a compatible CPU
kernel, and run its cells in order. P1–P12 have aligned reduced EEG and
kinematics files. The extra P55 files have unequal EEG and kinematic sample
counts and are not suitable for this pipeline as supplied. New results are
saved under `outputs/pipeline-v2-corrected-baseline-p<id>-<device>-<UTC timestamp>/`.
Model weights, raw prediction arrays, and the executed notebook are kept
locally by Git ignore rules; `results.json` and plots are publishable.
