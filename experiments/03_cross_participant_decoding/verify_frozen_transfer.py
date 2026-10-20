import csv
import hashlib
import json
from pathlib import Path
import sys
_repo = next(p for p in Path(__file__).resolve().parents if (p / "src/eeg_kinematics/paths.py").is_file())
sys.path.insert(0, str(_repo / "src"))
from eeg_kinematics.paths import REPO_ROOT, REDUCED_DATA_DIR
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

repo = REPO_ROOT
out = repo / 'outputs/frozen-p1-cross-participant-gpu-20260930T125956Z'
import argparse
parser = argparse.ArgumentParser(description='Verify saved frozen-transfer outputs without inference.')
parser.add_argument('--report-dir', type=Path)
args = parser.parse_args()
report_dir = (args.report_dir or out).resolve(); report_dir.mkdir(parents=True, exist_ok=True)

d = json.loads((out / 'results.json').read_text())
assert d['status'] == 'complete'
assert set(d['participants']) == {f'P{p}' for p in range(1,13)}
assert d['source_artifacts_unchanged'] and d['all_neural_weights_unchanged'] and d['P1_control_check'] == 'passed'
assert hashlib.sha256((out / 'experiment_source.py').read_bytes()).hexdigest() == d['experiment_sha256']
source=repo/d['source_directory']
for name, sha in d['source_artifact_sha256'].items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest() == sha
checks=0
windows=0
for label, participant in d['participants'].items():
    starts=loadmat(REDUCED_DATA_DIR/f'session_start_{label}.mat')['session_start'].ravel().astype(int)-1
    ends=np.r_[starts[1:],participant['provenance']['sample_count']]
    for name, entry in participant['models'].items():
        with np.load(out/f'{label}_{name}_predictions.npz') as saved:
            y,p,ids,targets=(saved[k] for k in ('y_true','y_pred','run_id','target_sample_index'))
            assert np.isfinite(y).all() and np.isfinite(p).all()
            if name=='training_mean': windows+=len(y)
            scopes={'primary_runs_8_9':np.isin(ids,[8,9]),'all_evaluated_runs':np.ones(len(ids),dtype=bool)}
            scopes.update({str(r):ids==r for r in np.unique(ids)})
            for scope, mask in scopes.items():
                expected=entry[scope] if scope in entry else entry['by_run'][scope]
                for metric, fn in [('mse',mean_squared_error),('mae',mean_absolute_error),('r2',r2_score)]:
                    np.testing.assert_allclose(fn(y[mask],p[mask],multioutput='raw_values'),expected[metric],rtol=1e-6,atol=1e-7)
                assert expected['n_windows']==int(mask.sum())
                checks+=1
            for run in np.unique(ids):
                expected=np.arange(starts[run-1],ends[run-1]-128,64)+128
                np.testing.assert_array_equal(targets[ids==run],expected)
assert len(list(csv.DictReader((out/'participant_metrics.csv').open())))==144
summary={'status':'passed','participants':12,'models':6,'saved_prediction_files':72,'metric_scopes_checked':checks,'windows_total':windows,
         'target_windows':windows-d['participants']['P1']['models']['training_mean']['all_evaluated_runs']['n_windows'],
         'source_hashes_match':True,'source_snapshot_hash_matches':True,'run_boundary_indices_match':True}
(report_dir/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
