# EEG-to-Kinematics Decoding

Research on decoding **wrist X/Y/Z position from 32-channel EEG** using [WAY-EEG-GAL](https://doi.org/10.1038/sdata.2014.47). The project preserves the original fork as a reference, corrects its evaluation pipeline, compares model variants, and investigates participant transfer, movement-stage errors and experimental conditions.

Model training so far centres on **P1**. Frozen-model transfer and dataset audits also cover **P2–P12**. Most model comparisons use one seed; the completed experiments do not establish population-level generalization.

## Experiment status

Updated **20 October 2026**. “Complete” refers to the stated scope, including experiments with negative findings. Full metrics, plots, protocols and limitations live in the linked reports.

| Family | Experiment | Status | Outcome / scope | Evidence |
|---|---|---|---|---|
| [00 · Dataset audits](experiments/00_dataset_audits/README.md) | Kinematic-target provenance | Complete | Raw/reduced matching identifies wrist X/Y/Z across P1–P12; exact upstream processing remains unresolved. | [Provenance](documentation/dataset/KINEMATICS_PROVENANCE.md) |
| 00 · Dataset audits | Vertical-axis verification | Complete | Position Z is vertical, positive upward. Force X is load/lift; force Z is grip. | [Axis audit](outputs/vertical-axis-audit-20261020/REPORT.md) |
| 00 · Dataset audits | Trial conditions and split coverage (C1) | Complete | Held-out current weight/surface combinations occur in training, but series-type composition differs. | [Condition audit](outputs/condition-audit-20261014T115618Z/RESULTS.md) |
| [01 · Continuous kinematic decoding](experiments/01_continuous_kinematic_decoding/README.md) | Original baseline v1, CPU/GPU | Complete · reference | Reproduction retained with its overlapping-window split and preprocessing limitations. | [CPU](outputs/original-p1-cpu/RESULTS.md), [GPU](outputs/original-p1-gpu/RESULTS.md) |
| 01 · Continuous kinematic decoding | Corrected baseline v2 | Complete · P1 | Whole-run splits, training-only scalers and independent validation establish the current reference protocol. | [v2 report](outputs/pipeline-v2-corrected-baseline-p1-gpu-20260929T173627Z/RESULTS.md) |
| [02 · Model comparison](experiments/02_model_comparison/README.md) | Six-model P1 comparison | Complete · single seed | ELU improves the original CNN; CNN–BiLSTM leads within P1 in this comparison. | [Six-model report](outputs/six-model-comparison-p1-gpu-20260930T095506Z/RESULTS.md) |
| [03 · Cross-participant decoding](experiments/03_cross_participant_decoding/README.md) | Frozen P1 models on P2–P12 | Complete | Poor transfer without adaptation; this evaluates one source participant, not population training. | [Transfer report](outputs/frozen-p1-cross-participant-gpu-20260930T125956Z/RESULTS.md) |
| [04 · Stage decoding](experiments/04_stage_decoding/README.md) | Saved-decoder movement-stage and error analyses (R1–R5) | Complete · initial P1 analysis | Event-aligned trajectories, stage errors, bias, density and low-speed sensitivity; no stage-specific training. | [Decoder analysis](outputs/saved-decoder-analysis-p1-20261013T172654Z/RESULTS.md) |
| 04 · Stage decoding | Trial averages and error timing | Complete · P1 runs 8–9 | Trial-average trajectories and exploratory simultaneous bands locate elevated error relative to preparation. | [Trial-average report](outputs/saved-decoder-analysis-p1-20261013T172654Z/trial_averages/RESULTS.md) |
| [05 · Condition decoding](experiments/05_condition_decoding/README.md) | Weight classification at fixed sandpaper (first C2 task) | Complete · initial P1 experiment | Logistic and CNN + ELU classifiers show no reliable held-out weight discrimination in this setup. | [Classification report](outputs/weight-classification-p1-gpu-20261014T115817Z/RESULTS.md) |
| [06 · Inference benchmark](experiments/06_inference_benchmark/README.md) | Saved-model CPU/GPU latency | Complete | Single-window and batch inference measured; acquisition and preprocessing are outside the timing scope. | [Latency report](outputs/inference-benchmark-p1-20261014/RESULTS.md) |

Stage-specific/cross-stage decoder training, fixed-condition versus pooled decoding, held-out-condition experiments, broader condition classification, and participant adaptation/population training remain **planned**. See the [project backlog](documentation/PROJECT_IDEAS.md) for individual ideas and status.

## Project navigation

| Location | Contents |
|---|---|
| [experiments/](experiments/README.md) | Numbered experiment families, entry points, commands and family READMEs. Both baseline notebooks live under `01_continuous_kinematic_decoding/`. |
| [src/eeg_kinematics/](src/eeg_kinematics) | Shared configurable data paths and resolution of historical source identities. |
| [configs/](configs/paths.example.toml) | Example data-path configuration; local overrides stay outside Git. |
| [documentation/](documentation/README.md) | Dataset definitions, methods, reproducibility guidance and research backlog. |
| [outputs/](outputs/README.md) | Saved execution reports, numerical results, plots and verification evidence. Weights and prediction arrays remain local. |
| [presentations/](presentations/README.md) | Links to native Google Slides decks and local export conventions. |
| [tests/](tests) and [scripts/](scripts/check_project_layout.py) | Path/notebook-startup tests and project navigation checks. |
| [CHANGELOG.md](CHANGELOG.md) | Version history and significant changes. |

## Data and coordinate definitions

The default workspace layout keeps datasets and reference papers outside this Git repository:

```text
EEG-EMG-Kinematics/
├── data/
│   ├── raw/way-eeg-gal/P1/ … P12/     # Original HS, WS and AllLifts files
│   └── external/way_eeg_clean/        # Downloaded reduced MAT files
├── references/papers/
└── hand_motion_eeg/                   # This repository
```

The baselines read `eeg_P<n>.mat`, `kin_P<n>.mat`, `session_start_P<n>.mat` and `P<n>_AllLifts.mat` from the reduced-data directory. The [Kaggle download](https://www.kaggle.com/datasets/radinkh2003/way-eeg-gal-clean) has partly undocumented upstream processing, so it is treated as external data. P1–P12 have aligned reduced recordings; the extra P55 files have mismatched sample counts.

Reduced target columns correspond to wrist `Px4`, `Py4`, `Pz4`. **Z is vertical (+ upward)**; X/Y anatomical directions remain unresolved. Reduced coordinates are normalized values, not physical distances. Position and contact-plate force axes use different frames. See [target provenance](documentation/dataset/KINEMATICS_PROVENANCE.md) and [coordinate definitions](documentation/dataset/COORDINATE_SYSTEM.md).

## Getting started

1. Use Python 3.9+ and install dependencies from the repository root:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Place data in the workspace locations above, or copy [configs/paths.example.toml](configs/paths.example.toml) to `configs/paths.local.toml` and set the locations. Environment-variable overrides and Windows/WSL guidance are documented in [layout and path configuration](documentation/reproducibility/PROJECT_LAYOUT.md).
3. Open the [corrected baseline v2 notebook](experiments/01_continuous_kinematic_decoding/corrected_baseline_v2.ipynb), set `PARTICIPANT_ID`, select the intended Python kernel, and run cells in order. For another experiment, use its family README in the status table.

The v2 reference uses runs **1–6 for training, 7 for validation, and 8–9 for testing**, with training-only EEG/target scalers. Input windows contain 128 EEG samples, advance by 64 samples and predict the next wrist sample without crossing run boundaries. [Full v2 protocol](documentation/methods/PIPELINE_V2_CORRECTED_BASELINE.md).

For repository checks:

```bash
python -m unittest discover -s tests
python scripts/check_project_layout.py
```

Fresh executions write separate result directories under `outputs/`. Saved-result verifiers accept `--report-dir` to write new checks without replacing published verification records. Existing executed snapshots retain their original provenance. See the [restructure verification](documentation/reproducibility/RESTRUCTURE_VERIFICATION.md).

## Sources

- Dataset: Luciw, Jarocka & Edin (2014), [Multi-channel EEG recordings during 3,936 grasp and lift trials with varying weight and friction](https://doi.org/10.1038/sdata.2014.47).
- Reduced data: [WAY-EEG-GAL Clean on Kaggle](https://www.kaggle.com/datasets/radinkh2003/way-eeg-gal-clean).
- Original fork: [Thowfiq23/hand_motion_eeg](https://github.com/Thowfiq23/hand_motion_eeg).
