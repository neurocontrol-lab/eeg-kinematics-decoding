# Stage decoding

Describe existing decoder errors by movement stage and timing, then assess the need for stage-specific training.

Saved-model diagnostics, trial averages and the first R6 dynamics analysis have reports below. The shared registry maintains execution status and priority.

Run commands from the repository root, with dependencies installed and data paths configured:

```bash
python experiments/04_stage_decoding/saved_model_analysis/analyze_saved_decoders.py
python experiments/04_stage_decoding/saved_model_analysis/average_saved_trials.py outputs/saved-decoder-analysis-p1-20261013T172654Z
python experiments/04_stage_decoding/saved_model_analysis/verify_saved_decoder_analysis.py outputs/saved-decoder-analysis-p1-20261013T172654Z
python experiments/04_stage_decoding/saved_model_analysis/verify_trial_averages.py outputs/saved-decoder-analysis-p1-20261013T172654Z
```

For fresh executions, choose a new output directory. Verification tools accept `--report-dir` to keep existing published verification records unchanged.

Methods: [documentation index](../../documentation/README.md). Data paths: [reproducibility guide](../../documentation/reproducibility/PROJECT_LAYOUT.md).

Saved results:
- [saved-decoder-analysis-p1-20261013T172654Z](../../outputs/saved-decoder-analysis-p1-20261013T172654Z/RESULTS.md)

## R6: movement dynamics and error

[Protocol](../../documentation/methods/ERROR_DYNAMICS.md) · [Commands and results](error_dynamics/README.md)
