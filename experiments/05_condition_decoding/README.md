# Condition decoding

Test whether EEG distinguishes experimental conditions. Weight classification is the first controlled task.

**Status:** P1 fixed-sandpaper weight classification completed. Surface and joint-condition classification remain open.

Run commands from the repository root, with dependencies installed and data paths configured:

```bash
python experiments/05_condition_decoding/weight_classification/classify_weight_p1.py --audit-dir outputs/condition-audit-20261014T115618Z
python experiments/05_condition_decoding/weight_classification/verify_weight_classification.py outputs/weight-classification-p1-gpu-20261014T115817Z
```

For fresh executions, choose a new output directory. Verification tools accept `--report-dir` to keep existing published verification records unchanged.

Methods: [documentation index](../../documentation/README.md). Data paths: [reproducibility guide](../../documentation/reproducibility/PROJECT_LAYOUT.md).

Saved results:
- [weight-classification-p1-gpu-20261014T115817Z](../../outputs/weight-classification-p1-gpu-20261014T115817Z/RESULTS.md)
