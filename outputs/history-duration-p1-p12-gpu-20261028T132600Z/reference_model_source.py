"""EXP-012: matched-target ridge/CNN + ELU timing study, validation-only selection."""
import os
for key,value in {'MPLBACKEND':'Agg','TF_FORCE_GPU_ALLOW_GROWTH':'true','TF_CPP_MIN_LOG_LEVEL':'2','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'}.items():os.environ.setdefault(key,value)
import argparse,csv,hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/eeg_kinematics/paths.py').is_file())
sys.path.insert(0,str(ROOT/'src'))
from eeg_kinematics.paths import REDUCED_DATA_DIR
from eeg_kinematics.timing_alignment import WINDOW,STRIDE,GAPS,matched_targets,make_windows
import numpy as np
import scipy,sklearn
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import tensorflow as tf
import matplotlib.pyplot as plt

PROTOCOL=ROOT/'documentation/methods/TIMING_ALIGNMENT.md'
PHASE_SOURCE=ROOT/'outputs/saved-decoder-analysis-p1-20261013T172654Z/trial_events.json'
REFERENCE=ROOT/'outputs/six-model-comparison-p1-gpu-20260930T095506Z/results.json'
ALPHAS=(1.,10.,100.,1000.,10000.)
STAGES=['unassigned/intertrial','preparation','reach','preload','loading','airborne_before_lowering_cue','after_lowering_cue','release','return']
SPLITS={'train':tuple(range(1,7)),'validation':(7,),'test':(8,9)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def metrics(y,p):
    assert y.shape==p.shape and len(y)>0 and np.isfinite(y).all() and np.isfinite(p).all()
    y,p=y.astype(float),p.astype(float);e=p-y;mse=np.mean(e*e,axis=0);ss=np.sum((y-y.mean(axis=0))**2,axis=0)
    return {'n':len(y),'mse':mse.tolist(),'mse_mean':float(mse.mean()),'rmse':np.sqrt(mse).tolist(),'mae':np.mean(abs(e),axis=0).tolist(),'r2':[float(1-np.sum(e[:,i]**2)/ss[i]) if ss[i]>1e-15 else None for i in range(3)],'mean_error_norm':float(np.linalg.norm(e,axis=1).mean())}
def build_cnn():
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input,Conv2D,BatchNormalization,DepthwiseConv2D,Activation,Dropout,SeparableConv2D,AveragePooling2D,Flatten,Dense
    model=Sequential([Input(shape=(128,32,1)),Conv2D(8,(64,1),padding='same',use_bias=False),BatchNormalization(),DepthwiseConv2D((1,32),use_bias=False,depth_multiplier=2),BatchNormalization(),Activation('elu'),Dropout(.25),SeparableConv2D(16,(16,1),padding='same',use_bias=False),BatchNormalization(),Activation('elu'),AveragePooling2D((4,1)),Dropout(.25),Flatten(),Dense(3)])
    model.compile(optimizer='adam',loss='mse',metrics=['mae']);assert model.count_params()==3235
    return model
def phases(indices,ids,bounds,gap):
    stage=np.zeros(len(indices),dtype=int);trial=np.zeros(len(indices),dtype=int);pure=np.zeros(len(indices),dtype=bool)
    events=json.loads(PHASE_SOURCE.read_text())
    for run in [8,9]:
        lo,hi=bounds[run];labels=np.zeros(hi-lo,dtype=int);trials=np.zeros(hi-lo,dtype=int)
        for item in events[str(run)]:
            b=np.rint(np.array(item['events'])*500).astype(int)
            assert np.all(np.diff(b)>=0) and b[0]>=0 and b[-1]<=len(labels)
            for s in range(1,9):labels[b[s-1]:b[s]]=s;trials[b[s-1]:b[s]]=item['trial']
        for j in np.flatnonzero(ids==run):
            t=int(indices[j]-lo);start=t-gap-127;stage[j]=labels[t];trial[j]=trials[t]
            pure[j]=bool(np.all(labels[start:t+1]==labels[t]) and np.all(trials[start:t+1]==trials[t]))
    return stage,trial,pure
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    out=(args.output_dir or ROOT/'outputs'/('timing-alignment-p1-gpu-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    assert out.parent==(ROOT/'outputs').resolve()
    if args.resume:assert (out/'results.json').exists()
    else:out.mkdir(exist_ok=False)
    print('OUTPUT_DIRECTORY',out,flush=True)
    sources={'runner':sha(__file__),'protocol':sha(PROTOCOL),'timing_helpers':sha(ROOT/'src/eeg_kinematics/timing_alignment.py'),'phase_events':sha(PHASE_SOURCE),'reference_results':sha(REFERENCE)}
    if args.resume:
        result=json.loads((out/'results.json').read_text());assert result['status']!='complete' and result['source_sha256']==sources
    else:
        result={'status':'running','phase':'fitting','experiment_id':'EXP-012','participant':'P1','created_at':stamp(),'seed':42,'sampling_rate':500,'window_samples':128,'stride_samples':64,'gap_samples':list(GAPS),'gap_ms':[g*2 for g in GAPS],'runs':{k:list(v) for k,v in SPLITS.items()},'source_sha256':sources,'scaling':{},'gaps':{},'selection':{},'test_evaluations':{},'limits':['Exploratory P1 single seed; historical test runs informed hypotheses.','Supplied reduced preprocessing is undocumented; causal indices do not establish causal preprocessing.','Ridge uses the existing bounded LSQR budget; candidate iteration counts are reported.','Overlapping windows are not independent replications.','Only validation selects gaps/alpha; test evaluates selected and baseline gaps only.','Stage R2 can be unstable for short intervals; stage results are descriptive.']}
        for source,name in [(Path(__file__),'experiment_source.py'),(PROTOCOL,'declared_protocol.md'),(ROOT/'src/eeg_kinematics/timing_alignment.py','timing_alignment_source.py')]: (out/name).write_bytes(source.read_bytes())
    def save(): (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    gpus=tf.config.list_physical_devices('GPU');assert gpus,'GPU required; no CPU fallback for CNN'
    for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
    with tf.device('/GPU:0'):probe=tf.matmul(tf.ones((32,32)),tf.ones((32,32)))
    assert 'GPU:0' in probe.device;probe.numpy()
    result['gpu_probe_device']=probe.device;result['gpu']=tf.config.experimental.get_device_details(gpus[0]);result['versions']={'python':sys.version.split()[0],'tensorflow':tf.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}
    eeg=np.asarray(loadmat(REDUCED_DATA_DIR/'eeg_P1.mat')['eeg'].T,dtype=np.float32,order='C');kin=np.asarray(loadmat(REDUCED_DATA_DIR/'kin_P1.mat')['kin'],dtype=np.float32,order='C')
    starts=loadmat(REDUCED_DATA_DIR/'session_start_P1.mat')['session_start'].ravel().astype(int)-1;ends=np.r_[starts[1:],len(eeg)];bounds={r:(int(lo),int(hi)) for r,(lo,hi) in enumerate(zip(starts,ends),1)}
    assert eeg.shape==(len(kin),32) and kin.shape[1]==3 and len(bounds)==9 and np.isfinite(eeg).all() and np.isfinite(kin).all()
    result['input_sha256']={name:sha(REDUCED_DATA_DIR/name) for name in ['eeg_P1.mat','kin_P1.mat','session_start_P1.mat']}
    es,ks=StandardScaler(),StandardScaler()
    for r in SPLITS['train']:
        lo,hi=bounds[r];es.partial_fit(eeg[lo:hi]);ks.partial_fit(kin[lo:hi])
    reference=json.loads(REFERENCE.read_text())['scaling']
    for key,scaler in [('eeg',es),('target',ks)]:
        np.testing.assert_array_equal(scaler.mean_,reference[key+'_mean']);np.testing.assert_array_equal(scaler.scale_,reference[key+'_scale'])
    result['scaling']={'fit_runs':list(SPLITS['train']),'eeg_mean':es.mean_.tolist(),'eeg_scale':es.scale_.tolist(),'target_mean':ks.mean_.tolist(),'target_scale':ks.scale_.tolist(),'samples_seen':int(es.n_samples_seen_),'identical_to_v2_reference':True}
    eeg=es.transform(eeg,copy=False);kin=ks.transform(kin,copy=False)
    splits={name:matched_targets(bounds,runs) for name,runs in SPLITS.items()};ys={name:kin[index] for name,(index,_) in splits.items()}
    result['split_shapes']={name:[len(index),128,32] for name,(index,_) in splits.items()};result['run_bounds']={str(r):list(b) for r,b in bounds.items()}
    np.savez_compressed(out/'split_indices.npz',**{name+'_'+key:value for name,(idx,ids) in splits.items() for key,value in [('targets',idx),('run_ids',ids)]})
    save();print('COMMON_TARGETS',result['split_shapes'],flush=True)
    for gap in GAPS:
        entry=result['gaps'].setdefault(str(gap),{'gap_ms':gap*2});print('GAP',gap*2,'ms',flush=True)
        if 'ridge' in entry and 'cnn_elu' in entry:continue
        train=make_windows(eeg,*splits['train'],bounds,gap);val=make_windows(eeg,*splits['validation'],bounds,gap)
        if 'ridge' not in entry:
            candidates=[]
            for alpha in ALPHAS:
                begin=time.perf_counter();ridge=Ridge(alpha=alpha,solver='lsqr',tol=1e-3,max_iter=200).fit(train.reshape(len(train),-1),ys['train'])
                pred=ridge.predict(val.reshape(len(val),-1));score=metrics(ys['validation'],pred);name=f'ridge_gap{gap}_alpha{alpha:g}.npz'
                np.savez_compressed(out/name,coef=ridge.coef_,intercept=ridge.intercept_,validation_pred=pred)
                candidate={'alpha':alpha,'validation':score,'n_iter':ridge.n_iter_.tolist(),'fit_seconds':time.perf_counter()-begin,'weights':name};candidates.append(candidate)
                print('RIDGE',gap*2,alpha,'validation MSE',score['mse_mean'],'iterations',candidate['n_iter'],flush=True)
            selected=min(candidates,key=lambda c:(c['validation']['mse_mean'],c['alpha']))
            entry['ridge']={'selected_alpha':selected['alpha'],'validation':selected['validation'],'weights':selected['weights'],'candidates':candidates};save()
        if 'cnn_elu' not in entry:
            tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42)
            with tf.device('/GPU:0'):
                cnn=build_cnn();init=hashlib.sha256(b''.join(w.tobytes() for w in cnn.get_weights())).hexdigest()
                if result.get('cnn_initial_sha256'):assert init==result['cnn_initial_sha256']
                else:result['cnn_initial_sha256']=init
                begin=time.perf_counter();hist=cnn.fit(train[...,None],ys['train'],validation_data=(val[...,None],ys['validation']),epochs=15,batch_size=64,verbose=2)
                fit_seconds=time.perf_counter()-begin;pred=cnn.predict(val[...,None],batch_size=64,verbose=0);name=f'cnn_elu_gap{gap}.keras';cnn.save(out/name)
            np.savez_compressed(out/f'cnn_elu_gap{gap}_validation.npz',y_pred=pred)
            entry['cnn_elu']={'validation':metrics(ys['validation'],pred),'fit_seconds':fit_seconds,'initial_sha256':init,'weights':name,'epochs':15,'parameters':3235,'history':{k:[float(v) for v in values] for k,values in hist.history.items()}}
            save();print('CNN_ELU',gap*2,'validation MSE',entry['cnn_elu']['validation']['mse_mean'],flush=True)
        del train,val
    # Freeze every choice before producing test predictions.
    for model in ['ridge','cnn_elu']:
        best=min(GAPS,key=lambda g:(result['gaps'][str(g)][model]['validation']['mse_mean'],g));result['selection'][model]={'gap_samples':best,'gap_ms':best*2,'validation_mse':result['gaps'][str(best)][model]['validation']['mse_mean']}
    result['selection_frozen_at']=result.get('selection_frozen_at') or stamp();result['phase']='testing';save()
    if not (out/'selection.json').exists():(out/'selection.json').write_text(json.dumps({'frozen_at':result['selection_frozen_at'],'models':result['selection'],'criterion':'minimum validation mean axis MSE; shorter-gap tie break'},indent=2)+'\n')
    else:assert json.loads((out/'selection.json').read_text())['models']==result['selection']
    print('SELECTION_FROZEN',result['selection'],flush=True)
    y=ys['test'];indices,ids=splits['test'];predictions={};pair_rows=[]
    for model in ['ridge','cnn_elu']:
        for gap in sorted({1,result['selection'][model]['gap_samples']}):
            key=f'{model}_gap{gap}';x=make_windows(eeg,indices,ids,bounds,gap)
            if key in result['test_evaluations']:
                with np.load(out/(key+'_predictions.npz')) as z:pred=z['y_pred']
            else:
                model_entry=result['gaps'][str(gap)][model]
                if model=='ridge':
                    with np.load(out/model_entry['weights']) as z:pred=x.reshape(len(x),-1)@z['coef'].T+z['intercept']
                else:
                    tf.keras.backend.clear_session();cnn=tf.keras.models.load_model(out/model_entry['weights'],compile=False)
                    with tf.device('/GPU:0'):pred=cnn.predict(x[...,None],batch_size=64,verbose=0)
                np.savez_compressed(out/(key+'_predictions.npz'),y_true=y,y_pred=pred,run_id=ids,target_sample_index=indices)
            stage,trial,pure=phases(indices,ids,bounds,gap)
            entry={'gap_samples':gap,'gap_ms':gap*2,'evaluated_at':result['test_evaluations'].get(key,{}).get('evaluated_at',stamp()),'aggregate':metrics(y,pred),'by_run':{str(r):metrics(y[ids==r],pred[ids==r]) for r in [8,9]},'by_stage':{},'pure_context_stage':{}}
            for s,label in enumerate(STAGES):
                for field,mask in [('by_stage',stage==s),('pure_context_stage',(stage==s)&pure)]:entry[field][label]=metrics(y[mask],pred[mask]) if mask.any() else {'n':0}
            result['test_evaluations'][key]=entry;predictions[key]=pred;save();del x
        baseline=predictions[f'{model}_gap1'];selected=predictions[f"{model}_gap{result['selection'][model]['gap_samples']}"]
        _,trials,_=phases(indices,ids,bounds,1)
        for run in [8,9]:
            for trial in range(1,35):
                mask=(ids==run)&(trials==trial);assert mask.any()
                b=float(np.linalg.norm(baseline[mask].astype(float)-y[mask],axis=1).mean());s=float(np.linalg.norm(selected[mask].astype(float)-y[mask],axis=1).mean())
                pair_rows.append({'model':model,'run':run,'trial':trial,'n_targets':int(mask.sum()),'baseline_mean_error_norm':b,'selected_mean_error_norm':s,'selected_minus_baseline':s-b})
    with (out/'paired_trial_errors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=pair_rows[0]);writer.writeheader();writer.writerows(pair_rows)
    result['paired_trials']={m:{'trials':68,'mean_selected_minus_baseline':float(np.mean([r['selected_minus_baseline'] for r in pair_rows if r['model']==m]))} for m in ['ridge','cnn_elu']}
    result['source_files_unchanged']=all(sha(p)==expected for p,expected in [(Path(__file__),sources['runner']),(PROTOCOL,sources['protocol']),(PHASE_SOURCE,sources['phase_events']),(REFERENCE,sources['reference_results'])]);assert result['source_files_unchanged']
    result['status']='complete';result['phase']='complete';result['completed_at']=stamp();save()
    fig,ax=plt.subplots(figsize=(8,4))
    for model,label in [('ridge','Ridge (validation-selected alpha)'),('cnn_elu','CNN + ELU (15 epochs)')]:ax.plot([g*2 for g in GAPS],[result['gaps'][str(g)][model]['validation']['mse_mean'] for g in GAPS],marker='o',label=label)
    ax.set(xlabel='Target minus final EEG input (ms)',ylabel='Validation MSE (standardized wrist targets)',title='P1 run 7: gap selection');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'validation_alignment.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for i,ax in enumerate(axes):
        keys=list(result['test_evaluations']);ax.bar(range(len(keys)),[result['test_evaluations'][k]['aggregate']['rmse'][i] for k in keys]);ax.set_xticks(range(len(keys)),[k.replace('_gap',' / gap samples ').replace('cnn_elu','CNN ELU') for k in keys],rotation=35,ha='right');ax.set(title='Wrist '+'XYZ'[i],ylabel='Test RMSE (standardized)')
    fig.tight_layout();fig.savefig(out/'selected_vs_baseline.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for gap,ax in zip(GAPS,axes.flat):
        hist=result['gaps'][str(gap)]['cnn_elu']['history'];ax.plot(range(1,16),hist['loss'],label='Train');ax.plot(range(1,16),hist['val_loss'],label='Validation');ax.set(title=f'Gap {gap*2} ms',xlabel='Epoch',ylabel='MSE');ax.legend()
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)
    gap=result['selection']['ridge']['gap_samples']
    with np.load(out/result['gaps'][str(gap)]['ridge']['weights']) as z:coef=z['coef'].reshape(3,128,32)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for i,ax in enumerate(axes):
        limit=max(float(abs(coef[i]).max()),1e-12);im=ax.imshow(coef[i].T,aspect='auto',origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit,extent=[-(gap+127)*2,-gap*2,.5,32.5]);ax.set(title='Wrist '+'XYZ'[i],xlabel='EEG time relative to target (ms)',ylabel='Channel index');fig.colorbar(im,ax=ax)
    fig.suptitle('Ridge prediction coefficients, not activation/source maps');fig.tight_layout();fig.savefig(out/'ridge_coefficients.png',dpi=160);plt.close(fig)
    lines=['# EXP-012: P1 EEG-to-kinematics timing alignment','',f"Completed {result['completed_at']}. Single participant and seed; exploratory reuse of historical test runs.",'','## Validation selection','','| End gap (ms) | Ridge validation MSE | Ridge alpha | CNN + ELU validation MSE |','|---|---:|---:|---:|']
    for gap in GAPS:
        e=result['gaps'][str(gap)];lines.append(f"| {gap*2} | {e['ridge']['validation']['mse_mean']:.6f} | {e['ridge']['selected_alpha']:g} | {e['cnn_elu']['validation']['mse_mean']:.6f} |")
    lines+=['','Lowest mean validation MSE chooses each model\'s gap; all choices froze before test evaluation. Every gap shares target timestamps. The current 2-ms reference was retrained on the common eligible set.','',f"Common windows: train {len(ys['train'])}, validation {len(ys['validation'])}, test {len(y)}. Train-only transforms match the existing v2 reference exactly.",'','## Permitted test comparisons','','| Model | Gap (ms) | Test MSE | R² X | R² Y | R² Z |','|---|---:|---:|---:|---:|---:|']
    for key,e in result['test_evaluations'].items():lines.append(f"| {key.rsplit('_gap',1)[0]} | {e['gap_ms']} | {e['aggregate']['mse_mean']:.6f} | "+' | '.join(f'{v:.4f}' if v is not None else 'undefined' for v in e['aggregate']['r2'])+' |')
    lines+=['','## Paired trial differences','','Negative selected-minus-baseline differences favor the validation-selected gap. These are descriptive, not independent-window significance tests.']
    for m,e in result['paired_trials'].items():lines.append(f"- {m}: {e['trials']} trials; mean selected-minus-baseline error norm {e['mean_selected_minus_baseline']:.6f} standardized units.")
    lines+=['','![Validation alignment](validation_alignment.png)','','![Selected and baseline test comparison](selected_vs_baseline.png)','','![Learning curves](learning_curves.png)','','![Ridge coefficients](ridge_coefficients.png)','','Per-run, target-stage and pure-context metrics are in results.json. Trial differences are in paired_trial_errors.csv. The exact executed protocol and source snapshots are retained. Weights/predictions stay local under Git ignore rules.','','[Declared protocol](../../documentation/methods/TIMING_ALIGNMENT.md) · [Registry](../../documentation/EXPERIMENT_REGISTRY.md)','','The experiment does not establish an optimal physiological delay, causal preprocessing, statistical significance, or participant generalization. No stage classifier, stage-specific regressor or condition-specific regressor was trained.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
