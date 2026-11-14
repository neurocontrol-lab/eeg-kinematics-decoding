# Movement dynamics and saved-decoder error

**Interactive results:** [Open the movement-stage report online](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/outputs/reports/movement-stages/index.html) · [Open locally](../../../outputs/reports/movement-stages/index.html)

R_6 asks whether observed position, event stages and movement dynamics
explain the errors of the six existing P1 decoders. It reuses their predictions;
the small ridge models predict error magnitude, not wrist position.

- [Declared protocol](../../../documentation/methods/ERROR_DYNAMICS.md)
- [P1 results](../../../outputs/error-dynamics-p1-20261028T091450Z/RESULTS.md)
- [Independent checks](../../../outputs/error-dynamics-p1-20261028T091450Z/verification.json)
- [Public registry](https://docs.google.com/spreadsheets/d/1Q2nJqr66ftuz7CP9dMiYRdE2RGwavmGbMGDkOS7yiWg/edit)

From the repository root in the existing scientific Python environment:

```bash
python experiments/04_stage_decoding/error_dynamics/analyse_error_dynamics.py
python experiments/04_stage_decoding/error_dynamics/verify_error_dynamics.py outputs/<new-R6-run>
```

The runner creates a new output directory and refuses to overwrite one. It
requires the local six-model predictions, verified R1–R5 traces/events and
reduced P1 wrist coordinates. It is CPU-only. Do not regenerate historical
predictions or overwrite prior source snapshots.

To re-render the documented summaries from saved numbers, without fitting:

```bash
python experiments/04_stage_decoding/error_dynamics/render_error_dynamics.py outputs/<R6-run>
```

The render operation records its own source and receipt separately from the
original execution snapshots. A completed result remains a retrospective,
exploratory two-series P1 analysis. Any new classifier, decoder or partitioning
study needs its own protocol.
