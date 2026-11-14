# EEG-to-kinematics timing alignment

**Interactive results:** [Open online](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/outputs/reports/timing-alignment/index.html) · [Open locally](../../outputs/reports/timing-alignment/index.html)

T_1: compare ridge and CNN + ELU at 2/50/100/200-ms gaps between final EEG input and wrist target. The 256-ms input window, 128-ms stride, v2 P1 run split and eligible target timestamps remain fixed. Ridge alpha and each model's gap are selected using validation MSE only. CNN requires GPU; ridge uses CPU.

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

Ridge selected 2 ms. CNN selected 100 ms on validation, but test MSE increased by 13.15% against the 2-ms common-target reference. The frozen selection is retained; no post-test retuning occurred. Phase effects differ, and the result remains exploratory P1/single seed. C_7 stage classification is next, with its protocol still to finalize.


## Population extension: T_8

[Declared population protocol](../../documentation/methods/POPULATION_TIMING_ALIGNMENT.md).
P1–P12 each receive ridge and CNN + ELU fits at every 2/50/100/200/500/1000-ms
end gap, using fixed 256-ms windows and shared eligible targets. CNN seeds
42/43/44 share one validation-selected gap per participant, selected by mean
validation MSE across seeds. All choices freeze before population test scoring.
This is within-participant replication across the dataset, not pooled or
unseen-participant training. Longer histories and temporal bins belong to the separately executed
T_2 and are outside this fixed-duration comparison.

```bash
python experiments/07_timing_alignment/run_population_timing.py --output-dir outputs/<new-population-run>
python experiments/07_timing_alignment/verify_population_timing.py outputs/<population-run>
```

Add `--resume` to the runner only for an interrupted run with unchanged source
and protocol. Candidate weights and predictions remain local under Git ignores.

### Memory-bounded resume

A WSL host-RAM exhaustion event interrupted T_8 during P2. Completed
checkpoints and all executed source hashes remain intact. The runtime queue
waits for the separate T2 job, resumes the unchanged runner, and releases its
owned process after each newly completed gap condition. No gaps, seeds, model
settings or selection rules change. The queue then runs saved-weight
verification, population reporting and independent summary verification.

```bash
python experiments/07_timing_alignment/run_population_queue.py --repo . --output-dir outputs/<population-run>
```

`pipeline_status.json` records waiting/training/verification status.
`training.log` retains subprocess output. The queue never stops the separate
T2 process. `verify_population_summary.py` checks participant aggregation and
bootstrap intervals from saved test predictions.

## Validation gap preferences across decoder architectures

[Protocol](../../documentation/methods/T1_DECODER_GAP_PREFERENCES.md).
CNN–BiLSTM and EEGNet regression use the same six gaps and participant-specific
T1 splits. The runner generates the validation-selected histograms and
per-participant scores after checking all saved validation predictions.

```bash
python experiments/07_timing_alignment/run_decoder_gap_preferences.py --output-dir outputs/timing-decoder-preferences-<timestamp>
```

Repeat the command with the same output directory to resume a stopped run.
