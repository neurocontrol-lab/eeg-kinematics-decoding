# EEG history duration and temporal binning

EXP-018 / T2 compares 256/512/1,024-ms histories as 128 channel-wise mean bins
for ridge and CNN + ELU. Each participant P1–P12 is trained separately; paired
participant effects form the group summary. This does not test a pooled decoder.

- [Declared protocol](../../documentation/methods/HISTORY_DURATION.md)
- [Public experiment registry](https://docs.google.com/spreadsheets/d/1Q2nJqr66ftuz7CP9dMiYRdE2RGwavmGbMGDkOS7yiWg/edit)

Run from the repository root in the existing GPU Python environment:

```bash
python experiments/08_history_duration/run_history_duration.py
python experiments/08_history_duration/summarize_history_duration.py outputs/<run>
python experiments/08_history_duration/verify_history_duration.py outputs/<run>
```

The runner requires a GPU for CNN fitting, refuses to overwrite completed runs,
and freezes all validation choices before test inference. Use `--output-dir`
with `--resume` only for an interrupted run with unchanged source/protocol.
Weights and prediction arrays remain local under existing Git ignore rules.

## Completed execution

[P1–P12 results](../../outputs/history-duration-p1-p12-gpu-20261028T132600Z/RESULTS.md) · [Group summary](../../outputs/history-duration-p1-p12-gpu-20261028T132600Z/group_summary.json) · [Independent verification](../../outputs/history-duration-p1-p12-gpu-20261028T132600Z/verification.json)

Both models and all three histories completed for 12 independently trained participants. The report gives paired participant effects, group R²/MSE and validation-selected history scores. Longer history and temporal averaging remain coupled.
