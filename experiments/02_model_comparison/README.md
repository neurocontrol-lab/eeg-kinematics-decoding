# Model comparison

Compare six P1 reference regressors under the corrected v2 protocol.

**Status:** Completed, single participant and seed. Training scripts require a working TensorFlow GPU.

Run commands from the repository root, with dependencies installed and data paths configured:

```bash
python experiments/02_model_comparison/compare_six_models.py
```

For fresh executions, choose a new output directory. Verification tools accept `--report-dir` to keep existing published verification records unchanged.

Methods: [documentation index](../../documentation/README.md). Data paths: [reproducibility guide](../../documentation/reproducibility/PROJECT_LAYOUT.md).

Saved results:
- [six-model-comparison-p1-gpu-20260930T095506Z](../../outputs/six-model-comparison-p1-gpu-20260930T095506Z/RESULTS.md)
