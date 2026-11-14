# Dataset audits

**Interactive results:** [Open online](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/outputs/reports/dataset-audits/index.html) · [Open locally](../../outputs/reports/dataset-audits/index.html)

Identify source channels, coordinate axes, trial conditions, and split coverage.

**Status:** Completed: target provenance, 3,528-trial condition audit, and 20 October vertical-axis audit.

Run commands from the repository root, with dependencies installed and data paths configured:

```bash
python experiments/00_dataset_audits/verify_kinematics.py
python experiments/00_dataset_audits/audit_trial_conditions.py
python experiments/00_dataset_audits/verify_vertical_axis.py --output outputs/vertical-axis-audit-<UTC-timestamp>
```

For fresh executions, choose a new output directory. Verification tools accept `--report-dir` to keep existing published verification records unchanged.

Methods: [documentation index](../../documentation/README.md). Data paths: [reproducibility guide](../../documentation/reproducibility/PROJECT_LAYOUT.md).

Saved results:
- [condition-audit-20261014T115618Z](../../outputs/condition-audit-20261014T115618Z/RESULTS.md)
- [vertical-axis-audit-20261020](../../outputs/vertical-axis-audit-20261020/REPORT.md)
