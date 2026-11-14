# Experiment index

[Browse all interactive reports online](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/outputs/reports/index.html) · [Local report index](../outputs/reports/index.html)

The public [experiment registry and roadmap](https://docs.google.com/spreadsheets/d/1Q2nJqr66ftuz7CP9dMiYRdE2RGwavmGbMGDkOS7yiWg/edit) owns experiment status, execution priority and run links. Update that Sheet directly; no local status table is maintained.

Stable numbers identify experiment families; C1–C7 and R1–R5 remain backlog identifiers. Execute commands from the repository root.

| Family | Question / scope | Interactive results |
|---|---|---|
| [00_dataset_audits](00_dataset_audits/README.md) | Identify source channels, coordinate axes, trial conditions, and split coverage. | [Open report](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/outputs/reports/dataset-audits/index.html) |
| [01_continuous_kinematic_decoding](01_continuous_kinematic_decoding/README.md) | Reproduce the original baseline and evaluate the corrected whole-run split. | [Open report](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/outputs/reports/continuous-decoding/index.html) |
| [02_model_comparison](02_model_comparison/README.md) | Compare six P1 reference regressors under the corrected v2 protocol. | [Open report](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/outputs/reports/model-comparison/index.html) |
| [03_cross_participant_decoding](03_cross_participant_decoding/README.md) | Evaluate frozen P1 models and P1 scalers on P2–P12 with zero adaptation. | [Open report](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/outputs/reports/participant-transfer/index.html) |
| [04_stage_decoding](04_stage_decoding/README.md) | Describe existing decoder errors by movement stage and timing, then assess the need for stage-specific training. | [Open report](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/outputs/reports/movement-stages/index.html) |
| [05_condition_decoding](05_condition_decoding/README.md) | Test whether EEG distinguishes experimental conditions. Weight classification is the first controlled task. | [Open report](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/outputs/reports/condition-decoding/index.html) |
| [06_inference_benchmark](06_inference_benchmark/README.md) | Measure warmed, synchronized CPU/GPU model-call latency. | [Open report](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/outputs/reports/inference-latency/index.html) |
| [07_timing_alignment](07_timing_alignment/README.md) | Compare input-to-target gaps with matched target timestamps and fixed model/window settings. | [Open report](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/outputs/reports/timing-alignment/index.html) |
| [08_history_duration](08_history_duration/README.md) | Compare 256/512/1,024-ms EEG histories as 128 temporal bins across independently trained P1–P12 decoders. | [Open report](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/outputs/reports/history-duration/index.html) |
| [09_movement_targets](09_movement_targets/README.md) | Compare wrist position and signed velocity decoding from matched EEG windows across P1–P12. | [Open report](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/outputs/reports/movement-targets/index.html) |

Shared path utilities live in `src/eeg_kinematics/`. Numerical models and preprocessing have not been extracted merely to create empty modules. Saved executions stay under [outputs](../outputs/README.md).
