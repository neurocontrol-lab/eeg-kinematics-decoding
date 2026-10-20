# Inference benchmark

Measure warmed, synchronized CPU/GPU model-call latency.

**Status:** Completed. Timing excludes acquisition, preprocessing, window accumulation, and model loading.

Run commands from the repository root, with dependencies installed and data paths configured:

```bash
python experiments/06_inference_benchmark/benchmark_inference.py --repo . --output outputs/inference-benchmark-<UTC-timestamp> --device cpu
```

For fresh executions, choose a new output directory. Verification tools accept `--report-dir` to keep existing published verification records unchanged.

Methods: [documentation index](../../documentation/README.md). Data paths: [reproducibility guide](../../documentation/reproducibility/PROJECT_LAYOUT.md).

Saved results:
- [inference-benchmark-p1-20261014](../../outputs/inference-benchmark-p1-20261014/RESULTS.md)
