# Continuous kinematic decoding

Reproduce the original baseline and evaluate the corrected whole-run split.

**Status:** Original v1 and corrected v2 retained. Numerical methods, split definitions, architectures, and epoch schedules are unchanged by the folder migration.

Run commands from the repository root, with dependencies installed and data paths configured:

```bash
jupyter notebook experiments/01_continuous_kinematic_decoding/corrected_baseline_v2.ipynb
```

For fresh executions, choose a new output directory. Verification tools accept `--report-dir` to keep existing published verification records unchanged.

Methods: [documentation index](../../documentation/README.md). Data paths: [reproducibility guide](../../documentation/reproducibility/PROJECT_LAYOUT.md).

Saved results:
- [original-p1-cpu](../../outputs/original-p1-cpu/RESULTS.md)
- [original-p1-gpu](../../outputs/original-p1-gpu/RESULTS.md)
- [pipeline-v2-corrected-baseline-p1-gpu-20260929T173627Z](../../outputs/pipeline-v2-corrected-baseline-p1-gpu-20260929T173627Z/RESULTS.md)

`original_baseline_v1.ipynb` retains the original numerical pipeline, including its evaluation limitations and historical Colab appendix. Setup/download cells now use configured local data. Saved executed notebooks and original hashes remain in their original output directories.
