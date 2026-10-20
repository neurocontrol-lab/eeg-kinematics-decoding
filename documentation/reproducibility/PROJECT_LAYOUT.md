# Workspace layout and path configuration

The workspace contains `data/raw/way-eeg-gal/P1` through `P12`, downloaded reduced files in `data/external/way_eeg_clean`, reference PDFs in `references/papers`, and this Git repository in `hand_motion_eeg`.

Raw and externally reduced data are local files outside Git. The reduced download is called external because its upstream processing is not fully documented. Data moves on 20 October 2026 preserved file identities, sizes and modification times. Existing output directories remain in place.

## Configure paths

Default paths derive from this repository's location, so the sibling workspace layout works in Windows and WSL. Override them with `configs/paths.local.toml` (copy `configs/paths.example.toml`) or environment variables `EEG_KINEMATICS_RAW_DIR` and `EEG_KINEMATICS_REDUCED_DIR`. Environment variables take precedence. `EEG_KINEMATICS_PATHS_CONFIG` can select another TOML file. Relative data paths resolve against the workspace, relative config-file paths against the repository. An explicitly selected missing config file is an error.

Use Python 3.9 or newer; Python versions before 3.11 use the conditional `tomli` dependency. Install dependencies with `python -m pip install -r requirements.txt`. The existing WSL GPU environment is suitable; GPU training is not necessary for path checks or saved-result verification.

Run scripts from the repository root using the commands in the [experiment index](../../experiments/README.md). A notebook can start from its own directory, the repository, or the workspace; it discovers the repository and imports the same shared paths. Notebook moves change active source hashes. Historical executed snapshots and their hashes are retained in saved outputs.

## Saved provenance and reports

Existing model weights, prediction arrays, numeric results, verification records, and executed source/protocol/notebook snapshots stay byte-for-byte unchanged. Only navigation links in human-readable result reports were updated. Old source identities such as `../P1/P1_AllLifts.mat` remain in historical JSON; `resolve_recorded_source()` maps these identities onto the configured data location. Do not rewrite those records to make them appear newly executed.

Verification commands accept `--report-dir <directory>` to write fresh checks outside the saved execution. Without this option they retain their original behaviour of writing beside the verified results. Future experiment executions use new output directories.

`presentations/README.md` indexes native Google Slides. Local deck exports and disposable builds are ignored by Git. Prior temporary helpers are archived under `tmp/legacy_workspace/`; those files are not supported experiment entry points. The retained original presentation builder requires the installed bundled runtime (`CODEX_PRESENTATION_RUNTIME`) and `PRESENTATIONS_SKILL_DIR`.

## Checks

```bash
python -m unittest discover -s tests
python scripts/check_project_layout.py
```

These checks cover configuration, notebook path discovery, current code syntax, local navigation links, and required data locations. Saved-result verification additionally checks the existing scientific evidence without retraining.
