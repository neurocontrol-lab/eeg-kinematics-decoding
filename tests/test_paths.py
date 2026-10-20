"""Check path configuration and notebook startup without loading EEG or TensorFlow."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from eeg_kinematics import paths

class PathsTests(unittest.TestCase):
    def test_defaults_do_not_depend_on_working_directory(self):
        saved=Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
                os.chdir(tmp)
                self.assertEqual(paths.data_path('raw'), (ROOT.parent/'data/raw/way-eeg-gal').resolve())
                self.assertEqual(paths.data_path('reduced'), (ROOT.parent/'data/external/way_eeg_clean').resolve())
                os.chdir(saved)
        finally:
            os.chdir(saved)

    def test_config_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'paths.toml'
            config.write_text('[data]\nraw="alternate/raw"\nreduced="alternate/reduced"\n',encoding='utf-8')
            with patch.dict(os.environ, {'EEG_KINEMATICS_PATHS_CONFIG':str(config)}, clear=True):
                self.assertEqual(paths.data_path('raw'), (ROOT.parent/'alternate/raw').resolve())
                with patch.dict(os.environ, {'EEG_KINEMATICS_RAW_DIR':str(Path(tmp)/'override')}):
                    self.assertEqual(paths.data_path('raw'), (Path(tmp)/'override').resolve())
                    self.assertEqual(paths.data_path('reduced'), (ROOT.parent/'alternate/reduced').resolve())

    def test_missing_explicit_config_is_an_error(self):
        with patch.dict(os.environ, {'EEG_KINEMATICS_PATHS_CONFIG':str(ROOT/'missing-paths.toml')}, clear=True):
            with self.assertRaises(FileNotFoundError):paths.data_path('raw')

    def test_historical_source_identities_resolve_to_relocated_data(self):
        self.assertEqual(paths.resolve_recorded_source('../P1/P1_AllLifts.mat'), paths.RAW_DATA_DIR/'P1/P1_AllLifts.mat')
        self.assertEqual(paths.resolve_recorded_source('way_eeg_clean/kin_P1.mat'), paths.REDUCED_DATA_DIR/'kin_P1.mat')
        self.assertEqual(paths.resolve_recorded_source('outputs/a/results.json'), ROOT/'outputs/a/results.json')

    def test_notebooks_find_paths_from_all_supported_start_directories(self):
        saved=Path.cwd()
        try:
            for notebook in [paths.ORIGINAL_NOTEBOOK, paths.CORRECTED_NOTEBOOK]:
                nb=json.loads(notebook.read_text(encoding='utf-8'))
                source=next(''.join(c['source']) for c in nb['cells'] if '_cwd = Path.cwd().resolve()' in ''.join(c.get('source',[])))
                begin=source.index('_cwd = Path.cwd().resolve()')
                end=source.index('assert DATA.is_dir()',begin)
                code='from pathlib import Path\nimport sys\n'+source[begin:end]
                for start in [ROOT, notebook.parent, ROOT.parent]:
                    os.chdir(start);scope={};exec(compile(code,str(notebook),'exec'),scope)
                    self.assertEqual(scope['PROJECT'],ROOT)
                    self.assertEqual(scope['DATA'],paths.REDUCED_DATA_DIR)
        finally:
            os.chdir(saved)

if __name__=='__main__':unittest.main()
