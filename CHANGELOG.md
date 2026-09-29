# Pipeline changelog

This is the central record of pipeline versions. Version names describe the evaluation protocol; they do not imply an improvement in model architecture or performance.

## v2 — Corrected baseline (2026-09-29)

[Notebook](pipeline_v2_corrected_baseline.ipynb) · [Detailed methods and limitations](documentation/PIPELINE_V2_CORRECTED_BASELINE.md) · [P1 GPU run](outputs/pipeline-v2-corrected-baseline-p1-gpu-20260929T173627Z/RESULTS.md)

- Added `PARTICIPANT_ID` at the top of the notebook. It selects the matching EEG, kinematics, run-boundary, and lift-metadata files and labels each new result. The reduced P1–P12 files align; the extra P55 EEG and kinematics files have different sample counts and fail the alignment check.
- Split complete recording runs before scaling or windowing. For each supplied nine-run participant, runs 1–6 train, run 7 validates, and runs 8–9 test.
- Fit separate EEG-channel and wrist-target scalers on training samples only. Keep each 128-sample, stride-64 window and its following target inside one run.
- Use validation data during training and evaluate test runs only after both fixed training schedules finish. Set seed 42 and report aggregate and per-test-run scores.
- Identified the three reduced kinematics columns as wrist X/Y/Z using the [source-data comparison](documentation/KINEMATICS_PROVENANCE.md).

The CNN and CNN–BiLSTM definitions and their 15/20-epoch schedules are unchanged from v1. The saved P1 GPU run reports CNN–BiLSTM R² of 0.784 / 0.757 / 0.726 for wrist X/Y/Z. P2 has passed loading, preprocessing, windowing, and model-build checks but has **not** been trained or evaluated. The P1 held-out runs contain friction condition 3 only; no cross-subject or population-model result has been produced.

## v1 — Original fork baseline

[Notebook](pipeline_v1_original.ipynb) · [Methods and limitations](documentation/ORIGINAL_BASELINE.md) · [P1 CPU run](outputs/original-p1-cpu/RESULTS.md) · [P1 GPU run](outputs/original-p1-gpu/RESULTS.md)

The original notebook uses overlapping windows with a random window-level split, scales data before the split, and passes the test partition to training as validation data. Its saved P1 runs reproduce that protocol. The v1 and v2 scores use different evaluation settings and target scaling, so they should not be interpreted as a direct model-performance comparison.
