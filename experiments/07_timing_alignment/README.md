# EEG-to-kinematics timing alignment

EXP-012 / T1: compare ridge and CNN + ELU at 2/50/100/200-ms gaps between final EEG input and wrist target. The 256-ms input window, 128-ms stride, v2 P1 run split and eligible target timestamps remain fixed. Ridge alpha and each model's gap are selected using validation MSE only. CNN requires GPU; ridge uses CPU.

- [Declared protocol](../../documentation/methods/TIMING_ALIGNMENT.md)
- [Public experiment registry and roadmap](https://docs.google.com/spreadsheets/d/1Q2nJqr66ftuz7CP9dMiYRdE2RGwavmGbMGDkOS7yiWg/edit)

From the repository root with the existing WSL GPU environment:

```bash
python experiments/07_timing_alignment/run_timing_alignment.py
```

The runner creates a fresh timestamped output directory. Resume an interrupted fitting run only with its original source/protocol using `--output-dir outputs/<run> --resume`. New source changes require a new experiment execution. Historical test runs have already informed the hypotheses, so this remains exploratory.

Verify a completed execution without fitting any model:

```bash
python experiments/07_timing_alignment/verify_timing_alignment.py outputs/<timing-run>
```

## Completed pilot

[Results and interpretation](../../outputs/timing-alignment-p1-gpu-20261027T120222Z/RESULTS.md) · [Independent checks](../../outputs/timing-alignment-p1-gpu-20261027T120222Z/verification.json)

Ridge selected 2 ms. CNN selected 100 ms on validation, but test MSE increased by 13.15% against the 2-ms common-target reference. The frozen selection is retained; no post-test retuning occurred. Phase effects differ, and the result remains exploratory P1/single seed. EXP-013 stage classification is next, with its protocol still to finalize.
