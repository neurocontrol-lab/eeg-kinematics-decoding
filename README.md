# EEG-to-Kinematics Decoding

Research on decoding **wrist X/Y/Z position from 32-channel EEG** using [WAY-EEG-GAL](https://doi.org/10.1038/sdata.2014.47). The project preserves the original fork as a reference, corrects its evaluation pipeline, compares model variants, and investigates participant transfer, movement-stage errors and experimental conditions.

The six-model references and initial stage analyses use **P1**. Timing, history-duration, and position-versus-velocity experiments train separate models for **P1–P12**. The main timing comparison repeats CNN + ELU training across three seeds. Frozen-model transfer and dataset audits also cover P2–P12. These within-participant studies do not establish performance on unseen participants.

## Experiment status

The public [experiment registry and roadmap](https://docs.google.com/spreadsheets/d/1Q2nJqr66ftuz7CP9dMiYRdE2RGwavmGbMGDkOS7yiWg/edit) is the single maintained record of experiment status, priorities, findings, dependencies and report links. Update the Sheet directly; this repository keeps protocols, code and saved evidence. Filter the registry by status or priority to review completed work and the next analyses.

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
3. Open the [corrected baseline v2 notebook](experiments/01_continuous_kinematic_decoding/corrected_baseline_v2.ipynb), set `PARTICIPANT_ID`, select the intended Python kernel, and run cells in order. For another experiment, use its family README in the [experiment index](experiments/README.md).

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
