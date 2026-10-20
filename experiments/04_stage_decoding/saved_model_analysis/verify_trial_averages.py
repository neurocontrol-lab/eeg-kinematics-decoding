"""Verify trial-average curves, coverage, corrected bands and unchanged sources."""
import argparse,csv,hashlib,json
from pathlib import Path
import sys
_repo = next(p for p in Path(__file__).resolve().parents if (p / "src/eeg_kinematics/paths.py").is_file())
sys.path.insert(0, str(_repo / "src"))
import numpy as np

p=argparse.ArgumentParser(description=__doc__);p.add_argument('analysis_dir',type=Path)
p.add_argument('--report-dir',type=Path)
args=p.parse_args();src=args.analysis_dir.resolve();out=src/'trial_averages'
report_dir=(args.report_dir or out).resolve();report_dir.mkdir(parents=True,exist_ok=True)
d=json.loads((out/'summary.json').read_text());a=np.load(out/'trial_arrays.npz')
assert d['status']=='complete' and d['source_files_unchanged']
assert d['bootstrap']['critical_value']>0 and d['minimum_coverage']==28
assert hashlib.sha256((out/'analysis_source.py').read_bytes()).hexdigest()==d['analysis_sha256']
for name,sha in d['source_sha256'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==sha
names=d['models'];grid=a['grid'];rows=list(csv.DictReader((out/'average_curves.csv').open()))
checks=0;norm_diff=[]
def mean(v):
 n=np.isfinite(v).sum(axis=0)
 return np.divide(np.nansum(v,axis=0),n,out=np.full(n.shape,np.nan),where=n>0)
for run in [8,9]:
 obs=a[f'run{run}_observed'];pred=a[f'run{run}_predicted'];error=a[f'run{run}_error'];base=a[f'run{run}_baseline']
 assert obs.shape[0]==34 and np.isfinite(base).all()
 np.testing.assert_allclose(np.linalg.norm(pred-obs,axis=-1),error,equal_nan=True)
 np.testing.assert_allclose(error-base[:,:,None],a[f'run{run}_difference'],equal_nan=True)
 coverage=np.isfinite(error).sum(axis=0)
 np.testing.assert_array_equal(coverage,a[f'run{run}_coverage'])
 for mi,name in enumerate(names):
  rr=[r for r in rows if int(r['run'])==run and r['model']==name]
  assert len(rr)==np.sum(coverage[mi]>0)
  y=mean(obs[:,mi]);yp=mean(pred[:,mi]);e=mean(error[:,mi]);de=mean(error[:,mi]-base[:,mi,None])
  norm_diff.append(float(np.nanmax(np.abs(e-np.linalg.norm(yp-y,axis=-1)))))
  for r in rr:
   j=int(round(float(r['seconds_from_LEDOn'])/.128));assert int(r['trials'])==coverage[mi,j]
   np.testing.assert_allclose([float(r['observed_'+v]) for v in 'xyz'],y[j],rtol=1e-10,atol=1e-12)
   np.testing.assert_allclose([float(r['predicted_'+v]) for v in 'xyz'],yp[j],rtol=1e-10,atol=1e-12)
   np.testing.assert_allclose(float(r['mean_error_norm']),e[j],rtol=1e-10,atol=1e-12)
   np.testing.assert_allclose(float(r['mean_paired_excess']),de[j],rtol=1e-10,atol=1e-12)
   eligible=r['eligible']=='True';sig=r['elevated']=='True'
   assert not eligible or coverage[mi,j]>=28
   assert sig==(eligible and float(r['simultaneous_low'])>0)
   if eligible:assert float(r['simultaneous_low'])<=de[j]<=float(r['simultaneous_high'])
   checks+=1
  assert sum(r['elevated']=='True' for r in rr)==d['results'][str(run)][name]['primary_significant_bins']
  for region in d['results'][str(run)][name]['regions']:
   selected=[r for r in rr if region['start_seconds_from_LEDOn']<float(r['seconds_from_LEDOn'])<region['end_seconds_from_LEDOn']]
   assert len(selected)==region['bins'] and all(r['elevated']=='True' for r in selected)
  assert (out/f'run_{run}_{name}_average.png').is_file()
assert set(names)=={'training_mean','ridge','cnn','cnn_bilstm','cnn_elu','eegnet_regression'}
assert {r['model'] for r in rows}==set(names)
assert len(list(out.glob('run_*_average.png')))==2*len(names)
assert (src/'run_8_trajectory.png').is_file() and (src/'trial_r8_t1_cnn_bilstm.png').is_file()
result={'status':'passed','runs':2,'models':len(names),'trials_per_run':34,'curve_rows_checked':checks,'source_hashes_match':True,
 'analysis_snapshot_hash_matches':True,'source_error_norms_and_trial_pairing_checked':True,'coverage_and_positive_lower_bound_checked':True,
 'original_plots_retained':True,'average_plots':2*len(names),'max_difference_between_mean_norm_and_norm_of_mean':max(norm_diff)}
(report_dir/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
