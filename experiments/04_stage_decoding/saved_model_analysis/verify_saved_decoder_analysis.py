from pathlib import Path
import sys
_repo = next(p for p in Path(__file__).resolve().parents if (p / "src/eeg_kinematics/paths.py").is_file())
sys.path.insert(0, str(_repo / "src"))
from eeg_kinematics.paths import REPO_ROOT
import argparse, csv, hashlib, json
import numpy as np
from sklearn.metrics import r2_score
root=REPO_ROOT
parser=argparse.ArgumentParser(description='Verify saved decoder-analysis outputs and add trial-level phase summaries.')
parser.add_argument('output_dir',type=Path)
parser.add_argument('--report-dir',type=Path)
args=parser.parse_args()
out=args.output_dir.resolve()
report_dir=(args.report_dir or out).resolve();report_dir.mkdir(parents=True,exist_ok=True)
assert out.parent==(root/'outputs').resolve()
d=json.loads((out/'results.json').read_text())
assert d['status']=='complete' and not d['trained_new_models']
assert sum(v['all_trial_EEG_and_wrist_sample_checks'] for v in d['clock_checks'].values())==68
assert all(v['valid_trials']==34 and not v['invalid_event_order_trials'] for v in d['clock_checks'].values())
assert hashlib.sha256((out/'analysis_source.py').read_bytes()).hexdigest()==d['analysis_sha256']
for name,h in d['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
rows=list(csv.DictReader((out/'aligned_predictions.csv').open()))
assert set(d['models'])=={'training_mean','ridge','cnn','cnn_bilstm','cnn_elu','eegnet_regression'}
assert {r['model'] for r in rows}==set(d['models'])
assert len(rows)==len(d['models'])*3642
checks=0;trial_stage=[];stage_summary={}
for name,entry in d['models'].items():
 rr=[r for r in rows if r['model']==name]
 y=np.array([[float(r['y_'+a]) for a in ['x','y','z']] for r in rr])
 p=np.array([[float(r['pred_'+a]) for a in ['x','y','z']] for r in rr])
 e=np.array([[float(r['error_'+a]) for a in ['x','y','z']] for r in rr])
 np.testing.assert_allclose(p-y,e,atol=1e-15)
 phases=np.array([r['stage'] for r in rr]);pure=np.array([r['pure_window']=='True' for r in rr])
 idx=np.array([int(r['target_index']) for r in rr]);assert np.all(np.diff(idx)>0)
 for label,mask in [('aggregate',np.ones(len(rr),bool))]+[(s,phases==s) for s in entry['by_stage']]:
  m=entry['aggregate'] if label=='aggregate' else entry['by_stage'][label]
  assert int(mask.sum())==m['n']
  if mask.any():
   np.testing.assert_allclose(np.mean(np.abs(e[mask]),axis=0),m['mae'],rtol=1e-9,atol=1e-12)
   np.testing.assert_allclose(np.mean(e[mask],axis=0),m['bias'],rtol=1e-9,atol=1e-12)
   np.testing.assert_allclose(np.mean(np.linalg.norm(e[mask],axis=1)),m['mean_euclidean_error'],rtol=1e-9,atol=1e-12)
  checks+=1
 for s in entry['by_stage_pure_window']:assert int(np.sum((phases==s)&pure))==entry['by_stage_pure_window'][s]['n']
 stage_summary[name]={}
 for stage in entry['by_stage']:
  values=[]
  for run in [8,9]:
   for trial in range(1,35):
    sel=np.array([int(r['run'])==run and int(r['trial'])==trial and r['stage']==stage for r in rr])
    if sel.any():
     value=float(np.mean(np.linalg.norm(e[sel],axis=1)));values.append(value)
     trial_stage.append([name,run,trial,stage,int(sel.sum()),value])
  if values:stage_summary[name][stage]={'trials_with_predictions':len(values),'median_trial_mean_error':float(np.median(values)),'q25':float(np.quantile(values,.25)),'q75':float(np.quantile(values,.75))}
with (report_dir/'trial_phase_errors.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['model','run','trial','stage','n','mean_euclidean_error']);w.writerows(trial_stage)
(report_dir/'trial_phase_summary.json').write_text(json.dumps(stage_summary,indent=2)+'\n')
result={'status':'passed','prediction_rows':len(rows),'models':len(d['models']),'aggregate_and_phase_metric_checks':checks,'trial_clock_checks':68,'pure_window_counts_checked':True,'source_prediction_hashes_match':True,'analysis_snapshot_hash_matches':True}
(report_dir/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
print('LEAD TRIAL PHASE SUMMARY',json.dumps(stage_summary['cnn_bilstm']))
