# Cross-participant decoding

Evaluate frozen P1 models and P1 scalers on P2–P12 with zero adaptation.

**Status:** Completed. This is single-source transfer, not leave-one-participant-out population training.

Run commands from the repository root, with dependencies installed and data paths configured:

```bash
python experiments/03_cross_participant_decoding/evaluate_frozen_p1.py
python experiments/03_cross_participant_decoding/verify_frozen_transfer.py
```

For fresh executions, choose a new output directory. Verification tools accept `--report-dir` to keep existing published verification records unchanged.

Methods: [documentation index](../../documentation/README.md). Data paths: [reproducibility guide](../../documentation/reproducibility/PROJECT_LAYOUT.md).

Saved results:
- [frozen-p1-cross-participant-gpu-20260930T125956Z](../../outputs/frozen-p1-cross-participant-gpu-20260930T125956Z/RESULTS.md)
