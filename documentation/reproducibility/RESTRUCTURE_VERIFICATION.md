# Restructure verification, 20 October 2026

The organization refactor changes paths and navigation, not model training or experimental protocols.

| Check | Outcome |
|---|---|
| Relocated data | All 292 dataset files retain the same file identity, byte length and modification time. |
| Existing saved executions | Checked all 245 existing output files. Numeric results, models, predictions, verification records and executed snapshots remain byte-identical. Only human-readable navigation/line endings changed in selected Markdown reports. |
| Path configuration | Five unit tests pass, covering working directories, config/environment precedence, missing explicit config, historical provenance identities, and both notebook startup locations. |
| Current navigation and syntax | Local Markdown/notebook links and Python source/cell syntax pass the project-layout checker. The presentation builder passes Node syntax checking; no deck was regenerated. |
| v2 preparation replay | Training/validation/test shapes remain 18,516/1,867/3,642 windows. Training scaler means/scales and saved test targets/run IDs/sample indices match exactly, starting from the notebook subdirectory. No training or neural inference. |
| Condition classifier verification | All 3,528 audit rows, 170 classifier trials, six saved probability files, 21 metric subsets, 2,312 circular offsets, source/protocol hashes and the recomputed training-only scaler pass. |
| Saved-decoder verification | All 21,852 trace rows, 60 aggregate/phase checks, 68 recorded trial-clock checks, pure-window counts and source/snapshot hashes pass. |
| Trial averages | All 534 curve rows, trial pairing, uncertainty-band flags, source hashes and 12 average plots pass. |
| Frozen transfer | All 72 prediction files, 750 metric scopes, 284,909 evaluated windows, source/snapshot hashes and run-boundary indices pass. |
| Vertical-axis evidence | The preserved executed source matches the saved runner SHA-256. The report and per-trial numeric evidence cover all 3,528 trials. No replacement experiment was claimed. |

Fresh verifier reports were written outside the saved executions. The current v1 notebook retains its historical Colab appendix and known scientific limitations; a complete v1 notebook execution and neural retraining were not part of this refactor.

Reproduction commands and current family entry points appear in the [experiment index](../../experiments/README.md). Use `--report-dir` with saved-result verifiers to retain the published records.
