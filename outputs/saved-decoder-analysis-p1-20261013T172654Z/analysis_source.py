"""Project Idea Category 2: Additional Results from Existing Decoders.
Chronological, event-aligned analysis of saved P1 decoders.
No training, prediction regeneration, threshold tuning on test labels, or source writes.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse, csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import correlate, savgol_filter
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'outputs/six-model-comparison-p1-gpu-20260930T095506Z'
NAMES=['training_mean','ridge','cnn','cnn_bilstm','cnn_elu','eegnet_regression']
STAGES=['unassigned/intertrial','preparation','reach','preload','loading','airborne_before_lowering_cue','after_lowering_cue','release','return']
EVENTS=['LEDOn','tHandStart','tFirstDigitTouch','tBothStartLoadPhase','tLiftOff','LEDOff','tReplace','tBothReleased','tHandStop']
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preserve-report',action='store_true',help='Keep the curated RESULTS.md when recomputing data and plots.')
parser.add_argument('--output-dir',type=Path,help='Recompute the analysis in an existing outputs directory.')
args=parser.parse_args()
out=(args.output_dir or ROOT/'outputs'/('saved-decoder-analysis-p1-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
assert out.parent==(ROOT/'outputs').resolve()
out.mkdir(exist_ok=args.output_dir is not None)
(out/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
meta=json.loads((SRC/'results.json').read_text())
sc=meta['scaling']
kin=loadmat(ROOT/'way_eeg_clean/kin_P1.mat',variable_names=['kin'])['kin']
starts=loadmat(ROOT/'way_eeg_clean/session_start_P1.mat')['session_start'].ravel().astype(int)-1
ends=np.r_[starts[1:],len(kin)]
p=loadmat(ROOT.parent/'P1/P1_AllLifts.mat',squeeze_me=True,struct_as_record=False)['P']
cols={str(n):i for i,n in enumerate(p.ColNames)}
info={'status':'running','participant':'P1','sources':[SRC.name],
 'scope':'Saved P1 held-out runs 8–9; six-model reference comparison',
 'trained_new_models':False,'coordinate_units':'Reduced target coordinates (0–1), not physical distances',
 'phase_assignment':'Target sample phase; pure-window subset requires every input sample and target in the same annotated phase',
 'clock_checks':{},'models':{},'limits':[
 'Descriptive held-out diagnostics, not a new model-selection experiment.',
 'Stage intervals use event landmarks; airborne_before_lowering_cue combines lifting and holding.',
 'after_lowering_cue includes reaction delay and lowering; event labels alone do not prove stationarity.',
 'Velocity uses the observed reduced wrist trajectory, in reduced-coordinate units per second.',
 'Density regions are observed marginal position bins, not proof of stationary behavior.',
 'Sparse 128-ms prediction cadence may undersample short preload/loading phases.',
 'These are pooled-trained decoders; no stage-specific decoder or cross-stage training matrix was fitted.',
 'Upstream filtering/scaling remain unknown; no claim about distinct neuron populations follows.']}

def metrics(y,z):
 if not len(y): return {'n':0}
 e=z-y
 return {'n':len(y),'mae':np.mean(np.abs(e),axis=0).tolist(),'rmse':np.sqrt(np.mean(e*e,axis=0)).tolist(),
         'bias':np.mean(e,axis=0).tolist(),'mean_euclidean_error':float(np.mean(np.linalg.norm(e,axis=1))),
         'r2':r2_score(y,z,multioutput='raw_values').tolist() if len(y)>1 else None}

def plain(o):
 if isinstance(o,dict):return {k:plain(v) for k,v in o.items()}
 if isinstance(o,list):return [plain(v) for v in o]
 if isinstance(o,float) and not np.isfinite(o):return None
 return o

def save(): (out/'results.json').write_text(json.dumps(plain(info),indent=2,allow_nan=False)+'\n')

raw={}; trials={}; phase_arrays={}; speed_arrays={}; fs=500
for run in [8,9]:
 h=loadmat(ROOT.parent/f'P1/HS_P1_S{run}.mat',squeeze_me=True,struct_as_record=False)['hs']
 w=loadmat(ROOT.parent/f'P1/WS_P1_S{run}.mat',squeeze_me=True,struct_as_record=False)['ws']
 assert h.eeg.samplingrate==h.kin.samplingrate==fs
 assert len(h.kin.sig)==ends[run-1]-starts[run-1]
 rows=p.AllLifts[p.AllLifts[:,cols['Run']]==run]
 assert len(rows)==len(w.win)
 # Infer the AllLifts/WS clock offset from exact EEG patterns, not an assumed origin.
 offsets=[]
 for j in [0,len(w.win)//2,len(w.win)-1]:
  q=w.win[j];v=q.eeg[:128,0].astype(float);x=h.eeg.sig[:,0].astype(float)
  cost=np.convolve(x*x,np.ones(len(v)),mode='valid')-2*correlate(x,v,mode='valid',method='fft')+np.sum(v*v)
  k=int(np.argmin(cost));np.testing.assert_array_equal(h.eeg.sig[k:k+128],q.eeg[:128])
  offsets.append(float(q.trial_start_time-k/fs))
 assert np.ptp(offsets)<1/fs
 offset=float(np.median(offsets)); checks=0
 for q,row in zip(w.win,rows):
  assert abs(row[cols['StartTime']]-q.trial_start_time)<1e-9
  k=round((q.trial_start_time-offset)*fs)
  np.testing.assert_array_equal(h.eeg.sig[k:k+100],q.eeg[:100])
  np.testing.assert_array_equal(h.kin.sig[k:k+100,[21,25,29]],q.kin[:100,[21,25,29]])
  checks+=1
 info['clock_checks'][str(run)]={'offset_seconds':offset,'all_trial_EEG_and_wrist_sample_checks':checks,
                                  'HS_samples':len(h.kin.sig),'sampling_rate':fs}
 arr=np.zeros(len(h.kin.sig),dtype=np.int8);tid=np.zeros(len(arr),dtype=int);valid=[];invalid=[]
 for j,row in enumerate(rows):
  ev=np.array([row[cols['StartTime']]+row[cols[k]]-offset for k in EVENTS])
  if not np.isfinite(ev).all() or np.any(np.diff(ev)<0) or ev[0]<0 or ev[-1]>len(arr)/fs:
   invalid.append(int(row[cols['Lift']]));continue
  bounds=np.rint(ev*fs).astype(int)
  for stage in range(1,9):arr[bounds[stage-1]:bounds[stage]]=stage;tid[bounds[stage-1]:bounds[stage]]=j+1
  valid.append({'trial':j+1,'events':ev.tolist(),'weight_code':int(row[cols['CurW']]),'surface_label':str(w.win[j].surf_id),'block_type':int(row[cols['BlockType']])})
 info['clock_checks'][str(run)].update({'valid_trials':len(valid),'invalid_event_order_trials':invalid})
 trials[run]=valid;phase_arrays[run]=(arr,tid)
 raw[run]=h.kin.sig[:,[21,25,29]]
 # Fixed 42-ms local polynomial smoothing; sensitivity below checks 22/82 ms.
 speed_arrays[run]={n:np.linalg.norm(savgol_filter(kin[starts[run-1]:ends[run-1]],n,2,deriv=1,delta=1/fs,axis=0),axis=1) for n in [11,21,41]}
 print('CLOCK_VERIFIED',run,offset,'trials',checks,flush=True)

# Density is defined on training-run observed coordinates, separately by axis.
edges=np.linspace(0,1,51);train=kin[:starts[6]];dense=[]
for a in range(3):
 counts,_=np.histogram(train[:,a],bins=edges);dense.append(np.argsort(-counts,kind='stable')[:5].tolist())
info['density_definition']={'training_runs':[1,2,3,4,5,6],'bins_per_axis':50,'dense_bin_count_per_axis':5,'dense_bins':dense,
 'note':'Top five training-occupancy bins per axis; report fractions in test, without using prediction errors to select regions.'}
info['speed_thresholds']=[.01,.02,.05]
model_paths=[(name,SRC,meta) for name in NAMES]
hashes={};target_meta=None;models={};phase_rows=[];sample_rows=[];trial_rows=[]
for name,source,m in model_paths:
 path=source/f'{name}_predictions.npz';hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
 z=np.load(path);idx=z['target_sample_index'];ids=z['run_id'];order=np.lexsort((idx,ids));idx=idx[order];ids=ids[order]
 assert set(ids)=={8,9} and len(np.unique(idx))==len(idx)
 yy=z['y_true'][order];yp=z['y_pred'][order]
 km=np.array(m['scaling']['target_mean']);ks=np.array(m['scaling']['target_scale'])
 y=yy*ks+km;pred=yp*ks+km
 np.testing.assert_allclose(y,kin[idx],rtol=1e-6,atol=1e-7)
 phases=np.empty(len(idx),dtype=int);tids=np.empty(len(idx),dtype=int);times=np.empty(len(idx));pure=np.zeros(len(idx),bool)
 for run in [8,9]:
  mask=ids==run;local=idx[mask]-starts[run-1];arr,ti=phase_arrays[run]
  np.testing.assert_array_equal(local,np.arange(128,len(arr),64))
  phases[mask]=arr[local];tids[mask]=ti[local];times[mask]=local/fs
  pure[mask]=[np.all(arr[t-128:t+1]==arr[t]) for t in local]
 e=pred-y;models[name]=(y,pred,idx,ids,times,phases,tids,pure)
 entry={'aggregate':metrics(y,pred),'by_stage':{},'by_stage_pure_window':{},'by_run':{},'motion_sensitivity':{},'density':{},'lag_diagnostic':{}}
 for si,stage in enumerate(STAGES):
  mask=phases==si;entry['by_stage'][stage]=metrics(y[mask],pred[mask]);entry['by_stage_pure_window'][stage]=metrics(y[mask&pure],pred[mask&pure])
  for subset,sel in [('all_targets',mask),('pure_window',mask&pure)]:
   mm=metrics(y[sel],pred[sel]);phase_rows.append([name,stage,subset,mm['n'],*mm.get('mae',[None]*3),*mm.get('rmse',[None]*3),*mm.get('bias',[None]*3),mm.get('mean_euclidean_error')])
 for run in [8,9]:
  mask=ids==run;entry['by_run'][str(run)]=metrics(y[mask],pred[mask])
  for tr in trials[run]:
   mask=(ids==run)&(tids==tr['trial']);mm=metrics(y[mask],pred[mask]);trial_rows.append([name,run,tr['trial'],tr['weight_code'],mm['n'],*mm.get('mae',[None]*3)])
 for smooth in [11,21,41]:
  speeds=np.concatenate([speed_arrays[r][smooth][idx[ids==r]-starts[r-1]] for r in [8,9]])
  entry['motion_sensitivity'][str(smooth)]={}
  for threshold in info['speed_thresholds']:
   stationary=speeds<=threshold
   entry['motion_sensitivity'][str(smooth)][str(threshold)]={'stationary_fraction':float(stationary.mean()),'stationary':metrics(y[stationary],pred[stationary]),'moving':metrics(y[~stationary],pred[~stationary])}
 main_speed=np.concatenate([speed_arrays[r][21][idx[ids==r]-starts[r-1]] for r in [8,9]])
 for a,axis in enumerate(['X','Y','Z']):
  bins=np.clip(np.searchsorted(edges,y[:,a],side='right')-1,0,49);inside=np.isin(bins,dense[a])
  entry['density'][axis]={'inside_fraction':float(inside.mean()),'inside_low_speed_fraction':float(np.mean(main_speed[inside]<=.02)) if inside.any() else None,
   'outside_low_speed_fraction':float(np.mean(main_speed[~inside]<=.02)) if (~inside).any() else None,
   'stage_counts_inside':{s:int(np.sum(inside&(phases==si))) for si,s in enumerate(STAGES)},
   'inside_mae':float(np.mean(np.abs(e[inside,a]))),'outside_mae':float(np.mean(np.abs(e[~inside,a])))}
  # Separate descriptive 2D scatter-density audit; this does not select a model.
  pe=np.linspace(-.5,1.5,101);hist,xe,ze=np.histogram2d(y[:,a],pred[:,a],bins=[edges,pe])
  cells=np.array(np.unravel_index(np.argsort(-hist.ravel(),kind='stable')[:10],hist.shape)).T
  ob=np.searchsorted(edges,y[:,a],side='right')-1;pb=np.searchsorted(pe,pred[:,a],side='right')-1
  valid=(y[:,a]>=0)&(y[:,a]<=1)&(pred[:,a]>=-.5)&(pred[:,a]<=1.5)
  ob=np.clip(ob,0,49);pb=np.clip(pb,0,99)
  dense2=valid & np.any((ob[:,None]==cells[:,0])&(pb[:,None]==cells[:,1]),axis=1)
  entry['density'][axis]['scatter_2d']={'definition':'Ten most occupied held-out observed/predicted cells, 0.02-wide bins; descriptive only',
   'dense_cells':cells.tolist(),'inside_fraction':float(dense2.mean()),'out_of_grid_count':int((~valid).sum()),
   'inside_low_speed_fraction':float(np.mean(main_speed[dense2]<=.02)) if dense2.any() else None,
   'inside_mean_absolute_error':float(np.mean(np.abs(e[dense2,a]))) if dense2.any() else None,
   'outside_mean_absolute_error':float(np.mean(np.abs(e[~dense2,a]))) if (~dense2).any() else None,
   'stage_counts_inside':{s:int(np.sum(dense2&(phases==si))) for si,s in enumerate(STAGES)}}
  lag=[]
  for shift in range(-4,5):
   ya=[];pa=[]
   for run in [8,9]:
    q=ids==run;v=y[q,a];u=pred[q,a]
    if shift>=0:ya.extend(v[shift:]);pa.extend(u[:len(u)-shift])
    else:ya.extend(v[:shift]);pa.extend(u[-shift:])
   lag.append({'shift_seconds':shift*64/fs,'correlation':float(np.corrcoef(ya,pa)[0,1]) if np.std(pa)>1e-12 else None})
  entry['lag_diagnostic'][axis]={'shifts':lag,'note':'Positive shift compares prediction at t with observation at t+shift. Test-only descriptive scan; not an optimized model lag.'}
 for j in range(len(idx)):sample_rows.append([name,int(ids[j]),int(tids[j]),int(idx[j]),float(times[j]),STAGES[phases[j]],bool(pure[j]),*y[j],*pred[j],*e[j],float(main_speed[j])])
 info['models'][name]=entry
 print('ANALYZED',name,flush=True)

def csvwrite(name,header,rows):
 with (out/name).open('w',newline='') as f:
  w=csv.writer(f);w.writerow(header);w.writerows(rows)
csvwrite('phase_metrics.csv',['model','stage','subset','n','mae_x','mae_y','mae_z','rmse_x','rmse_y','rmse_z','bias_x','bias_y','bias_z','mean_euclidean_error'],phase_rows)
csvwrite('trial_metrics.csv',['model','run','trial','weight_code','n','mae_x','mae_y','mae_z'],trial_rows)
csvwrite('aligned_predictions.csv',['model','run','trial','target_index','run_time_seconds','stage','pure_window','y_x','y_y','y_z','pred_x','pred_y','pred_z','error_x','error_y','error_z','speed'],sample_rows)
(out/'trial_events.json').write_text(json.dumps(trials,indent=2)+'\n')
info['source_sha256']=hashes
colors=plt.cm.Set3(np.linspace(0,1,9))
for run in [8,9]:
 for tr in [trials[run][0],trials[run][len(trials[run])//2],trials[run][-1]]:
  for name in ['cnn_bilstm','cnn_elu','eegnet_regression']:
   y,pred,idx,ids,times,phases,tids,pure=models[name];ev=np.array(tr['events']);sel=(ids==run)&(times>=max(0,ev[0]-.3))&(times<ev[-1]+.3)
   fig,ax=plt.subplots(3,2,figsize=(14,8),sharex=True,layout='constrained')
   for a in range(3):
    for col in range(2):
     for si in range(1,9):ax[a,col].axvspan(ev[si-1]-ev[0],ev[si]-ev[0],color=colors[si],alpha=.3)
     for t in ev:ax[a,col].axvline(t-ev[0],color='gray',lw=.5)
    ax[a,0].plot(times[sel]-ev[0],y[sel,a],'k.-',label='Observed');ax[a,0].plot(times[sel]-ev[0],pred[sel,a],'.-',label='Predicted')
    ax[a,1].plot(times[sel]-ev[0],pred[sel,a]-y[sel,a],'.-',color='tab:red');ax[a,1].axhline(0,color='black',lw=.7)
    ax[a,0].set_ylabel('XYZ'[a]+' (reduced)');ax[a,1].set_ylabel('Prediction − observation');ax[a,0].legend(loc='upper right')
   for a in ax[-1]:a.set_xlabel('Seconds from LEDOn')
   ax[0,0].set_title('Observed and predicted');ax[0,1].set_title('Signed error')
   fig.suptitle(f'P1 run {run}, trial {tr["trial"]}: {name}')
   fig.legend(handles=[Patch(facecolor=colors[si],alpha=.6,label=STAGES[si].replace('_',' ')) for si in range(1,9)],loc='lower center',bbox_to_anchor=(.5,-.08),ncol=4,fontsize=9)
   fig.savefig(out/f'trial_r{run}_t{tr["trial"]}_{name}.png',dpi=140,bbox_inches='tight');plt.close(fig)
 # Continuous time course for the lead model, including intertrial samples.
 y,pred,idx,ids,times,phases,tids,pure=models['cnn_bilstm'];sel=ids==run
 fig,ax=plt.subplots(4,1,figsize=(15,9),sharex=True,layout='constrained')
 for a in range(3):ax[a].plot(times[sel],y[sel,a],label='Observed',color='black',lw=.8);ax[a].plot(times[sel],pred[sel,a],label='Predicted',lw=.7);ax[a].set_ylabel('XYZ'[a]);ax[a].legend()
 ax[3].plot(times[sel],np.linalg.norm(pred[sel]-y[sel],axis=1),color='tab:red');ax[3].set_ylabel('Error norm');ax[3].set_xlabel('Run time (seconds)')
 fig.suptitle(f'P1 run {run}: CNN–BiLSTM chronological reconstruction (reduced coordinates)');fig.savefig(out/f'run_{run}_trajectory.png',dpi=150);plt.close(fig)

fig,ax=plt.subplots(figsize=(14,6),layout='constrained');xx=np.arange(len(STAGES));width=.12
for j,name in enumerate(NAMES):
 vals=[info['models'][name]['by_stage'][s].get('mean_euclidean_error',np.nan) for s in STAGES]
 ax.bar(xx+(j-2.5)*width,vals,width,label=name)
ax.set_xticks(xx,[s.replace('_','\n') for s in STAGES],fontsize=9);ax.set_ylabel('Mean Euclidean error (reduced coordinates)');ax.set_title('Pooled-trained models: error across event-defined phases');ax.legend(ncol=3)
fig.savefig(out/'phase_error_comparison.png',dpi=160);plt.close(fig)
from plot_stage_scatter import plot_stage_scatter
y,pred,*_=models['cnn_bilstm'];phases=models['cnn_bilstm'][5]
plot_stage_scatter(y,pred,phases,STAGES,out/'scatter_by_stage.png')
assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
info['analysis_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
info['source_predictions_unchanged']=True;info['status']='complete';save()
lines=['# Project Idea Category 2: Additional Results from Existing Decoders','',
 'P1 analysis of R1–R5 from the [project ideas backlog](../../documentation/PROJECT_IDEAS.md).','',
 'No models were trained or predictions regenerated. Six P1 reference models use held-out runs 8–9.',
 'R1 reconstructs saved target samples; R2 plots coordinates and signed error versus time; R3 reports bias/phase errors and a descriptive lag scan; R4 traces scatter-density regions; R5 compares low-speed and moving samples.','',
 '## Alignment and stage definitions','',
 'Raw HS and WS EEG and wrist samples match exactly for all 68 held-out trials. The AllLifts/WS time origin is 2 seconds later than the continuous HS origin in these files; the offset is inferred from exact EEG matching and validated for every trial.',
 'Times on plots use the verified continuous clock. This conversion must not be replaced by blindly adding StartTime to trial-relative events.',
 'Stages are preparation, reach, preload, loading, airborne before lowering cue, after lowering cue, release, and return. Airborne includes both lifting/holding; the cue interval includes reaction time. Unassigned/intertrial samples are retained and not automatically called rest.',
 'Phase refers to the target sample. Separate pure-window metrics require all 128 EEG samples and the target to share that phase. Short stages can have few or no pure windows.','',
 '## Phase error (mean Euclidean distance in reduced coordinates)','',
 '**Time reference: LEDOn = 0 s in each trial.** Typical intervals use median event times across 68 trials. Metrics use each trial\'s actual boundaries, not the median bins. Intervals are start-inclusive/end-exclusive, in seconds (0.25 s = 250 ms).','',
 '| Stage | Event boundaries | Typical interval (s from LEDOn) | Samples | CNN–BiLSTM | CNN + ELU | EEGNet |','|---|---|---|---:|---:|---:|---:|']
typical_events=np.median(np.array([np.array(t['events'])-t['events'][0] for ts in trials.values() for t in ts]),axis=0)
boundary_names=['Outside annotated LEDOn → hand-stop intervals','LEDOn → hand start','Hand start → first digit touch','First digit touch → load-phase onset on both digits','Load-phase onset on both digits → lift-off','Lift-off → LEDOff','LEDOff → object replacement','Object replacement → both digits released','Both digits released → hand stop']
for stage in STAGES:
 vals=[info['models'][n]['by_stage'][stage] for n in ['cnn_bilstm','cnn_elu','eegnet_regression']]
 si=STAGES.index(stage)
 interval=f'[{typical_events[si-1]:.3f}, {typical_events[si]:.3f})' if si else 'No single trial-relative interval'
 lines.append(f'| {stage} | {boundary_names[si]} | {interval} | {vals[0]["n"]} | '+' | '.join(f'{v.get("mean_euclidean_error",float("nan")):.4f}' for v in vals)+' |')
lines+=['','Exact boundaries are in trial_events.json; subtract each trial\'s LEDOn time for its own relative intervals. Unassigned samples are not restricted to a fixed pre-event window such as [-0.25, 0) s.']
lines+=['','![Phase errors](phase_error_comparison.png)','','![Run 8 trajectory](run_8_trajectory.png)','','![Example trial](trial_r8_t1_cnn_bilstm.png)','','![Stage-colored scatter](scatter_by_stage.png)','','## Stationary/moving and density diagnostics','',
 'Velocity comes from the observed reduced coordinates with fixed local polynomial smoothing. Primary exploratory settings: 42-ms smoothing, speed threshold 0.02 reduced units/s. Sensitivity covers 22/42/82-ms smoothing and thresholds 0.01/0.02/0.05. These are descriptive definitions, not validated physical rest labels.',
 'Dense observed-position regions are the five most occupied bins in each axis, using 50 bins over 0–1 and training runs 1–6 only. A separate descriptive scatter audit uses the ten most occupied held-out 2D observed/predicted cells with 0.02-wide bins. Both definitions and their fractions, speed composition, stage counts and inside/outside errors are in results.json. Neither definition itself proves resting positions.','',
 '## Files and limits','',
 'aligned_predictions.csv retains source target indices, verified times, phases, observed/predicted coordinates, errors and speed. phase_metrics.csv contains all-target/pure-window errors, bias and per-axis metrics; trial_metrics.csv contains individual-trial summaries. results.json includes sensitivity analyses and lag diagnostics. trial_events.json records event boundaries.',
 'Plots use saved predictions spaced 128 ms apart. Lines connect decoded samples for viewing; they do not provide additional decoded time points. Errors use reduced coordinates, not physical distance. Stage-wise R² can be unstable when target variance is small.',
 'Stage comparisons here use pooled-trained decoders. They identify difficult intervals but cannot yet establish the benefit of a stage-specific decoder or distinct neural populations. A cross-stage training/evaluation matrix is a subsequent C5 experiment.',
 'Event metadata source: https://github.com/luciw/way-eeg-gal-utilities/blob/master/Utilities/WEEG_GetEventsInHS.m. The local files additionally require the empirically verified time-origin conversion.','']
if not args.preserve_report:
 (out/'RESULTS.md').write_text('\n'.join(lines))
lead=info['models']['cnn_bilstm']
findings=['','## Initial findings for the six-model CNN–BiLSTM','',
 'Mean error is higher during reach (0.3014) and return (0.2525) than preparation (0.0589) and the airborne interval (0.1016), in reduced-coordinate Euclidean units. These are descriptive pooled-model errors, not evidence that separate decoders are already necessary.',
 'Pure-window counts matter: preload has none; loading has 26 and release only four. Phase labels based on targets can otherwise mix earlier-stage EEG into a later-stage prediction.',
 f'At the primary exploratory speed threshold, {100*lead["motion_sensitivity"]["21"]["0.02"]["stationary_fraction"]:.1f}% of targets are low speed. Low-speed mean error is 0.0778 versus 0.1464 for moving targets. This does not establish that most samples are at rest.',
 'The aggregate signed Y bias is +0.0633 reduced units; X and Z biases are −0.0038 and −0.0073. The time plots show whether these offsets persist during particular stages.','',
 '| Speed threshold (reduced units/s, 42-ms smoothing) | Low-speed fraction | Low-speed error | Moving error |',
 '|---|---:|---:|---:|']
for threshold in info['speed_thresholds']:
 v=lead['motion_sensitivity']['21'][str(threshold)]
 findings.append(f'| {threshold} | {100*v["stationary_fraction"]:.1f}% | {v["stationary"]["mean_euclidean_error"]:.4f} | {v["moving"]["mean_euclidean_error"]:.4f} |')
findings+=['','| Axis | Dense 2D-scatter fraction | Low-speed fraction within dense cells | MAE inside | MAE outside |','|---|---:|---:|---:|---:|']
for axis in ['X','Y','Z']:
 v=lead['density'][axis]['scatter_2d']
 findings.append(f'| {axis} | {100*v["inside_fraction"]:.1f}% | {100*v["inside_low_speed_fraction"]:.1f}% | {v["inside_mean_absolute_error"]:.4f} | {v["outside_mean_absolute_error"]:.4f} |')
findings+=['','Next: inspect these traces and phase/sample coverage before choosing C5 stage-specific training. Any difficult-stage claim should control for target variance, condition, trial composition and window overlap with preceding phases.','']
if not args.preserve_report:
 with (out/'RESULTS.md').open('a') as f:f.write('\n'.join(findings))
print('COMPLETE',out,flush=True)
