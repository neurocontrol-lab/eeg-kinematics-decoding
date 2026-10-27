"""Independently verify EXP-012 saved evidence; no training or model selection changes."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,csv,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/eeg_kinematics/paths.py').is_file());sys.path.insert(0,str(ROOT/'src'))
from eeg_kinematics.paths import REDUCED_DATA_DIR
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import tensorflow as tf

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_metrics(y,p,e):
    assert y.shape==p.shape and len(y)==e['n'] and np.isfinite(p).all()
    mse=mean_squared_error(y.astype(float),p.astype(float),multioutput='raw_values')
    np.testing.assert_allclose(mse,e['mse'],rtol=1e-10,atol=1e-12);np.testing.assert_allclose(np.sqrt(mse),e['rmse'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(mse.mean(),e['mse_mean'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(mean_absolute_error(y.astype(float),p.astype(float),multioutput='raw_values'),e['mae'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(p.astype(float)-y.astype(float),axis=1).mean(),e['mean_error_norm'],rtol=1e-10,atol=1e-12)
    for axis in range(3):
        if e['r2'][axis] is not None:np.testing.assert_allclose(r2_score(y[:,axis].astype(float),p[:,axis].astype(float)),e['r2'][axis],rtol=1e-9,atol=1e-11)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output_dir',type=Path);args=parser.parse_args();out=args.output_dir.resolve();assert out.parent==(ROOT/'outputs').resolve()
    d=json.loads((out/'results.json').read_text());assert d['status']=='complete' and d['experiment_id']=='EXP-012' and d['source_files_unchanged']
    for name,key in [('experiment_source.py','runner'),('declared_protocol.md','protocol'),('timing_alignment_source.py','timing_helpers')]:assert sha(out/name)==d['source_sha256'][key]
    phase_path=ROOT/'outputs/saved-decoder-analysis-p1-20261013T172654Z/trial_events.json';assert sha(phase_path)==d['source_sha256']['phase_events']
    for name,digest in d['input_sha256'].items():assert sha(REDUCED_DATA_DIR/name)==digest
    assert d['gap_samples']==[1,25,50,100] and d['window_samples']==128 and d['stride_samples']==64
    assert d['runs']=={'train':list(range(1,7)),'validation':[7],'test':[8,9]}
    eeg=np.asarray(loadmat(REDUCED_DATA_DIR/'eeg_P1.mat')['eeg'].T,dtype=np.float32,order='C');kin=np.asarray(loadmat(REDUCED_DATA_DIR/'kin_P1.mat')['kin'],dtype=np.float32,order='C')
    starts=loadmat(REDUCED_DATA_DIR/'session_start_P1.mat')['session_start'].ravel().astype(int)-1;ends=np.r_[starts[1:],len(eeg)];assert len(starts)==9
    es,ks=StandardScaler(),StandardScaler()
    for lo,hi in zip(starts[:6],ends[:6]):es.partial_fit(eeg[lo:hi]);ks.partial_fit(kin[lo:hi])
    for name,scaler in [('eeg',es),('target',ks)]:
        np.testing.assert_array_equal(scaler.mean_,d['scaling'][name+'_mean']);np.testing.assert_array_equal(scaler.scale_,d['scaling'][name+'_scale'])
    assert es.n_samples_seen_==d['scaling']['samples_seen']==sum(ends[:6]-starts[:6])
    eeg=es.transform(eeg,copy=False);kin=ks.transform(kin,copy=False)
    splits={};indices_file=np.load(out/'split_indices.npz')
    for name,runs in d['runs'].items():
        all_t=[];all_r=[]
        for run in runs:
            lo,hi=starts[run-1],ends[run-1]
            for t in range(int(lo)+128,int(hi),64):
                if all(t-gap-127>=lo for gap in [1,25,50,100]):all_t.append(t);all_r.append(run)
        t=np.asarray(all_t);r=np.asarray(all_r)
        np.testing.assert_array_equal(t,indices_file[name+'_targets']);np.testing.assert_array_equal(r,indices_file[name+'_run_ids']);assert d['split_shapes'][name]==[len(t),128,32]
        splits[name]=(t,r)
    assert [len(splits[k][0]) for k in ['train','validation','test']]==[18504,1865,3638]
    def features(t,gap):return np.array([eeg[i-gap-127:i-gap+1] for i in t],dtype=np.float32)
    gpu=tf.config.list_physical_devices('GPU');assert gpu,'GPU required for saved CNN replay checks'
    for g in gpu:tf.config.experimental.set_memory_growth(g,True)
    validation_checks=0;replay_checks=0
    for gap in [1,25,50,100]:
        entry=d['gaps'][str(gap)];t,_=splits['validation'];x=features(t,gap);y=kin[t]
        candidates=entry['ridge']['candidates'];assert [c['alpha'] for c in candidates]==[1.,10.,100.,1000.,10000.]
        for c in candidates:
            with np.load(out/c['weights']) as z:
                predicted=x.reshape(len(x),-1)@z['coef'].T+z['intercept'];np.testing.assert_allclose(predicted,z['validation_pred'],rtol=1e-5,atol=1e-5);check_metrics(y,z['validation_pred'],c['validation'])
            assert all(0<i<=200 for i in c['n_iter']);validation_checks+=1
        selected=min(candidates,key=lambda c:(c['validation']['mse_mean'],c['alpha']));assert selected['alpha']==entry['ridge']['selected_alpha'] and selected['weights']==entry['ridge']['weights'];assert selected['validation']==entry['ridge']['validation']
        with np.load(out/f'cnn_elu_gap{gap}_validation.npz') as z:
            saved=z['y_pred'];check_metrics(y,saved,entry['cnn_elu']['validation'])
        assert entry['cnn_elu']['epochs']==15 and entry['cnn_elu']['parameters']==3235 and entry['cnn_elu']['initial_sha256']==d['cnn_initial_sha256']
        model=tf.keras.models.load_model(out/entry['cnn_elu']['weights'],compile=False)
        with tf.device('/GPU:0'):predicted=model.predict(x[...,None],batch_size=64,verbose=0)
        np.testing.assert_allclose(predicted,saved,rtol=1e-3,atol=5e-4);replay_checks+=1;validation_checks+=1;del model,x
        tf.keras.backend.clear_session()
    choices=json.loads((out/'selection.json').read_text());assert choices['models']==d['selection'] and choices['frozen_at']==d['selection_frozen_at']
    for model in ['ridge','cnn_elu']:
        best=min([1,25,50,100],key=lambda g:(d['gaps'][str(g)][model]['validation']['mse_mean'],g));assert best==d['selection'][model]['gap_samples']
    permitted={f'{m}_gap{g}' for m in ['ridge','cnn_elu'] for g in {1,d['selection'][m]['gap_samples']}}
    assert set(d['test_evaluations'])==permitted and {p.name.replace('_predictions.npz','') for p in out.glob('*_predictions.npz')}==permitted
    t,ids=splits['test'];y=kin[t];phases=np.zeros(len(t),int);trial=np.zeros(len(t),int);segments=[];events=json.loads(phase_path.read_text());labels=['unassigned/intertrial','preparation','reach','preload','loading','airborne_before_lowering_cue','after_lowering_cue','release','return']
    for run in [8,9]:
        lo=int(starts[run-1]);cursor=lo
        for item in events[str(run)]:
            b=[lo+round(v*500) for v in item['events']]
            if cursor<b[0]:segments.append((run,cursor,b[0],0,0))
            for phase,(begin,end) in enumerate(zip(b[:-1],b[1:]),1):segments.append((run,begin,end,phase,item['trial']))
            cursor=b[-1]
        segments.append((run,cursor,int(ends[run-1]),0,0))
    for run,begin,end,phase,number in segments:
        mask=(ids==run)&(t>=begin)&(t<end);phases[mask]=phase;trial[mask]=number
    predictions={};metric_checks=0
    for key,e in d['test_evaluations'].items():
        model,gap=key.rsplit('_gap',1);gap=int(gap);assert e['evaluated_at']>=d['selection_frozen_at']
        with np.load(out/(key+'_predictions.npz')) as z:
            np.testing.assert_array_equal(z['target_sample_index'],t);np.testing.assert_array_equal(z['run_id'],ids);np.testing.assert_array_equal(z['y_true'],y);pred=z['y_pred']
        check_metrics(y,pred,e['aggregate']);metric_checks+=1
        for run in [8,9]:mask=ids==run;check_metrics(y[mask],pred[mask],e['by_run'][str(run)]);metric_checks+=1
        x=features(t,gap)
        if model=='ridge':
            with np.load(out/d['gaps'][str(gap)][model]['weights']) as z:replay=x.reshape(len(x),-1)@z['coef'].T+z['intercept']
        else:
            cnn=tf.keras.models.load_model(out/d['gaps'][str(gap)][model]['weights'],compile=False)
            with tf.device('/GPU:0'):replay=cnn.predict(x[...,None],batch_size=64,verbose=0)
        np.testing.assert_allclose(replay,pred,rtol=1e-3 if model=='cnn_elu' else 1e-5,atol=5e-4 if model=='cnn_elu' else 1e-5);replay_checks+=1;del x
        pure=np.zeros(len(t),bool)
        for run,begin,end,phase,number in segments:pure|=(ids==run)&(t<end)&(t-gap-127>=begin)
        for i,label in enumerate(labels):
            for field,mask in [('by_stage',phases==i),('pure_context_stage',(phases==i)&pure)]:
                assert int(mask.sum())==e[field][label]['n']
                if mask.any():check_metrics(y[mask],pred[mask],e[field][label]);metric_checks+=1
        predictions[key]=pred
    pairs=list(csv.DictReader((out/'paired_trial_errors.csv').open()));assert len(pairs)==136
    for row in pairs:
        m=row['model'];mask=(ids==int(row['run']))&(trial==int(row['trial']));assert mask.sum()==int(row['n_targets'])
        baseline=predictions[m+'_gap1'];selected=predictions[f"{m}_gap{d['selection'][m]['gap_samples']}"]
        b=np.linalg.norm(baseline[mask].astype(float)-y[mask].astype(float),axis=1).mean();s=np.linalg.norm(selected[mask].astype(float)-y[mask].astype(float),axis=1).mean()
        np.testing.assert_allclose([b,s,s-b],[float(row[k]) for k in ['baseline_mean_error_norm','selected_mean_error_norm','selected_minus_baseline']],rtol=1e-10,atol=1e-12)
    for m in ['ridge','cnn_elu']:
        rr=[r for r in pairs if r['model']==m];assert len(rr)==68;np.testing.assert_allclose(np.mean([float(r['selected_minus_baseline']) for r in rr]),d['paired_trials'][m]['mean_selected_minus_baseline'],atol=1e-12)
    for name in ['RESULTS.md','validation_alignment.png','selected_vs_baseline.png','learning_curves.png','ridge_coefficients.png']:assert (out/name).is_file() and (out/name).stat().st_size>0
    report={'status':'passed','checked_at':datetime.now(timezone.utc).isoformat(),'source_input_and_protocol_hashes_match':True,'train_only_scalers_recomputed':True,'common_targets_and_run_boundaries_checked':True,'validation_metric_sets':validation_checks,'saved_weight_replay_checks':replay_checks,'permitted_test_configurations':len(permitted),'test_metric_subsets_checked':metric_checks,'paired_trial_rows_checked':136,'validation_only_gap_and_alpha_selection':True,'identical_cnn_initializations_recorded':True,'models_trained_by_verifier':0}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
