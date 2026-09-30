# EEG-to-Kinematics Decoding

Research baselines for decoding movement kinematics from EEG in the [WAY-EEG-GAL dataset](https://doi.org/10.1038/sdata.2014.47). The current notebooks predict the **wrist tracker's X, Y, and Z position** from 32-channel EEG. This repository was originally forked from [Thowfiq23/hand_motion_eeg](https://github.com/Thowfiq23/hand_motion_eeg); it preserves that pipeline as v1 and adds a separate, evaluation-corrected v2. The present results are within-participant baselines, not evidence of cross-participant decoding.

## Pipelines and repository layout

| Path | Purpose |
| --- | --- |
| [`pipeline_v1_original.ipynb`](pipeline_v1_original.ipynb) | Original fork pipeline, retained as a reproduction reference. Its random overlapping-window split and preprocessing limit what its scores establish. |
| [`pipeline_v2_corrected_baseline.ipynb`](pipeline_v2_corrected_baseline.ipynb) | Current baseline. Selects a participant at the top, holds out complete recording runs, fits scalers on training runs, and evaluates test runs after training. It retains the v1 CNN and CNN–BiLSTM architectures. |
| [`CHANGELOG.md`](CHANGELOG.md) | Central record of changes between pipeline versions. |
| [`documentation/`](documentation/) | [V1 methods](documentation/ORIGINAL_BASELINE.md), [v2 methods](documentation/PIPELINE_V2_CORRECTED_BASELINE.md), and [kinematics-target provenance](documentation/KINEMATICS_PROVENANCE.md). |
| [`outputs/`](outputs/) | [Published run index](outputs/README.md), numeric results, and plots. Raw predictions, model weights, and executed notebooks stay local. |
| [`requirements.txt`](requirements.txt) | Python dependencies; exact versions used by a saved run are recorded in its `results.json`. |

## Data and prediction target

The notebooks use the [cleaned WAY-EEG-GAL files on Kaggle](https://www.kaggle.com/datasets/radinkh2003/way-eeg-gal-clean). Download them separately and place the MAT files directly under `way_eeg_clean/` at the repository root. This directory is ignored by Git.

For participant `P<n>`, v2 reads:

| File | Role |
| --- | --- |
| `eeg_P<n>.mat` | `eeg`: 32 channels × samples; transposed to samples × channels in the notebook. |
| `kin_P<n>.mat` | `kin`: samples × 3 wrist-position targets. |
| `session_start_P<n>.mat` | One-based start sample of each recording run; converted to zero-based boundaries. |
| `P<n>_AllLifts.mat` | Lift-trial metadata, including run and friction-condition labels. |

The reduced files for **P1–P12** each contain nine aligned runs and can be selected in v2. The extra P55 EEG and kinematics files have unequal sample counts; v2 stops with an alignment error for that participant. A [sample-aligned comparison](documentation/KINEMATICS_PROVENANCE.md) with original P1 and P6 recordings identifies the three reduced target columns as wrist position X, Y, and Z (`Px4`, `Py4`, `Pz4`). Their exact upstream filtering, scaling formula, and physical units in the reduced files are not documented.

## Run the corrected baseline

1. Create a compatible Python environment and install dependencies with `python -m pip install -r requirements.txt`.
2. Put the downloaded MAT files directly in `way_eeg_clean/`.
3. Open [`pipeline_v2_corrected_baseline.ipynb`](pipeline_v2_corrected_baseline.ipynb) from the repository root in Jupyter or VS Code. Set `PARTICIPANT_ID = 1` in its first code cell to the participant you want, then run cells in order. The [v2 methods](documentation/PIPELINE_V2_CORRECTED_BASELINE.md) include the local CPU and WSL GPU kernel notes.

The notebook writes a participant- and device-labeled directory such as `outputs/pipeline-v2-corrected-baseline-p1-gpu-<UTC timestamp>/`. Its `RESULTS.md`, `results.json`, and plots are publishable; model weights, raw prediction arrays, and execution snapshots are excluded from Git.

### V2 evaluation protocol

- For the supplied nine-run participants, runs **1–6 train**, run **7 validates**, and runs **8–9 test**. No run appears in more than one split.
- Separate `StandardScaler` instances fit EEG channels and wrist targets using **training samples only**.
- Each input has **128 EEG samples**; starts advance by **64 samples** within a run. The target is the immediately following wrist-position sample. Windows and targets never cross a run boundary.
- The CNN and CNN–BiLSTM keep the original model definitions and fixed 15/20-epoch schedules. The test set is evaluated only after both fits finish. Scores are reported overall and per test run.

## Published results and interpretation

The first [v2 P1 GPU run](outputs/pipeline-v2-corrected-baseline-p1-gpu-20260929T173627Z/RESULTS.md) produced:

| Model | Wrist X R² | Wrist Y R² | Wrist Z R² |
| --- | ---: | ---: | ---: |
| CNN | 0.547 | 0.399 | 0.451 |
| CNN–BiLSTM | 0.784 | 0.757 | 0.726 |

These scores are for held-out runs **of P1**, not an unseen participant. Both P1 test runs contain friction condition 3 only. P2 has passed data-loading, preprocessing, windowing, and model-build checks but has **not** been trained or evaluated. The [v1 CPU/GPU runs](outputs/README.md) use a different split and target scaling, so their scores are not a direct comparison of model quality with v2. MSE and MAE in v2 use standardized target units; R² is unchanged by that affine target scaling.

## Next research steps

Repeat run-level evaluation with predeclared splits covering other friction conditions; then establish a population model and evaluate participant-specific fine-tuning without mixing target-participant test windows into training. Cross-laboratory or cross-dataset transfer remains future work. See the [changelog](CHANGELOG.md) and [v2 limitations](documentation/PIPELINE_V2_CORRECTED_BASELINE.md) before extending the pipeline.

## Sources

- [Luciw, Jarocka, and Edin, *Multi-channel EEG recordings during 3,936 grasp and lift trials with varying weight and friction*](https://doi.org/10.1038/sdata.2014.47).
- [Cleaned WAY-EEG-GAL dataset used by the notebooks](https://www.kaggle.com/datasets/radinkh2003/way-eeg-gal-clean).
- [Original fork source](https://github.com/Thowfiq23/hand_motion_eeg).
