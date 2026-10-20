"""Portable project paths; no data loading or directory creation on import."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent

def _configured_data():
    config = Path(os.environ.get("EEG_KINEMATICS_PATHS_CONFIG", REPO_ROOT / "configs/paths.local.toml")).expanduser()
    if not config.is_absolute():
        config = REPO_ROOT / config
    if not config.is_file():
        if "EEG_KINEMATICS_PATHS_CONFIG" in os.environ:
            raise FileNotFoundError(config)
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with config.open("rb") as stream:
        return tomllib.load(stream).get("data", {})

def data_path(kind):
    defaults = {"raw": "data/raw/way-eeg-gal", "reduced": "data/external/way_eeg_clean"}
    if kind not in defaults:
        raise ValueError(f"Unknown data kind: {kind}")
    value = os.environ.get(f"EEG_KINEMATICS_{kind.upper()}_DIR") or _configured_data().get(kind, defaults[kind])
    path = Path(value).expanduser()
    return (path if path.is_absolute() else WORKSPACE_ROOT / path).resolve()

RAW_DATA_DIR = data_path("raw")
REDUCED_DATA_DIR = data_path("reduced")
OUTPUTS_DIR = REPO_ROOT / "outputs"
BASELINE_DIR = REPO_ROOT / "experiments/01_continuous_kinematic_decoding"
ORIGINAL_NOTEBOOK = BASELINE_DIR / "original_baseline_v1.ipynb"
CORRECTED_NOTEBOOK = BASELINE_DIR / "corrected_baseline_v2.ipynb"

def resolve_recorded_source(recorded):
    """Resolve legacy source identities without rewriting saved provenance records."""
    path = str(recorded).replace(chr(92), "/")
    if path.startswith("../P"):
        return RAW_DATA_DIR / path[3:]
    if path.startswith("raw/P"):
        return RAW_DATA_DIR / path[4:]
    if path.startswith("way_eeg_clean/"):
        return REDUCED_DATA_DIR / path.split("/", 1)[1]
    return REPO_ROOT / path
