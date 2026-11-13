"""No-fitting verification of EXP-042 indexing, scalers, selections and predictions."""
import os
for k,v in {'TF_CPP_MIN_LOG_LEVEL':'2','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','TF_FORCE_GPU_ALLOW_GROWTH':'true'}.items():os.environ.setdefault(k,v)
import argparse,hashlib,json,sys,gc
from pathlib import Path
from datetime import datetime,timezone
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/eeg_kinematics/paths.py').is_file())
sys.path.insert(0,str(ROOT/'src'))
from eeg_kinematics.paths import REDUCED_DATA_DIR,RAW_DATA_DIR
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import tensorflow as tf
from run_timing_alignment import build_cnn

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for a in iter(lambda:f.read(2**20),b''):h.update(a)
    return h.hexdigest()

def check(y,p,m):
    y=y.astype(float);p=p.astype(float)
    assert y.shape==p.shape and len(y)==m['n'] and np.isfinite(p).all()
    e=mean_squared_error(y,p,multioutput='raw_values')
    for expected,actual in [(e,m['mse']),(e.mean(),m['mse_mean']),(np.sqrt(e),m['rmse']),(mean_absolute_error(y,p,multioutput='raw_values'),m['mae']),(np.linalg.norm(y-p,axis=1).mean(),m['mean_error_norm'])]:
        np.testing.assert_allclose(expected,actual,rtol=1e-10,atol=1e-12)
    for axis in range(3):
        if m['r2'][axis] is not None:np.testing.assert_allclose(r2_score(y[:,axis],p[:,axis]),m['r2'][axis],rtol=1e-9,atol=1e-11)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output_dir',type=Path);a=ap.parse_args();out=a.output_dir.resolve()
    assert out.parent==(ROOT/'outputs').resolve()
    d=json.loads((out/'results.json').read_text());assert d['status']=='complete' and d['experiment_id']=='EXP-042' and d['source_files_unchanged']
    assert d['participants']==[f'P{i}' for i in range(1,13)] and d['gap_samples']==[1,25,50,100,250,500] and d['cnn_seeds']==[42,43,44]
    assert (d['window_samples'],d['stride_samples'],d['sampling_hz'])==(128,64,500)
    assert d['splits']=={'train':list(range(1,7)),'validation':[7],'test':[8,9]}
    sources={'runner':'runner_source.py','protocol':'declared_protocol.md','pilot_runner':'pilot_runner_source.py','timing_helper':'timing_helper_source.py'}
    for key,file in sources.items():assert sha(out/file)==d['source_sha256'][key]
    audit_path=ROOT/'outputs/condition-audit-20261014T115618Z/results.json';assert sha(audit_path)==d['source_sha256']['condition_audit']
    audit=json.loads(audit_path.read_text());assert audit['status']=='complete'
    frozen=json.loads((out/'selection.json').read_text());assert frozen['participants']==d['selection'] and frozen['frozen_at']==d['selection_frozen_at']
    gpus=tf.config.list_physical_devices('GPU');assert gpus
    for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
    initial={}
    for seed in [42,43,44]:
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed);model=build_cnn()
        initial[str(seed)]=hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest();del model
    counts={'participants_checked':0,'input_files_hashed':0,'validation_metric_sets_checked':0,'ridge_validation_weight_replays':0,'cnn_validation_weight_replays':0,'test_weight_replays':0,'test_metric_sets_checked':0,'ridge_axis_fits_at_iteration_cap':0}
    for person in d['participants']:
        print('VERIFY_PARTICIPANT',person,flush=True)
        e=d['participant_results'][person];directory=out/person
        assert e['cnn_initial_hashes']==initial
        raw=RAW_DATA_DIR/person/f'{person}_AllLifts.mat'
        assert sha(raw)==audit['source_sha256'][f'../{person}/{person}_AllLifts.mat']==d['condition_audit'][person]['metadata_sha256']
        for r in (7,8,9):assert audit['participants'][person]['runs'][str(r)]['series_type']=='weight' and audit['participants'][person]['runs'][str(r)]['surfaces']==[3]
        for name,digest in e['input_sha256'].items():assert sha(REDUCED_DATA_DIR/name)==digest;counts['input_files_hashed']+=1
        eeg=np.asarray(loadmat(REDUCED_DATA_DIR/f'eeg_{person}.mat')['eeg'].T,dtype=np.float32,order='C')
        y=np.asarray(loadmat(REDUCED_DATA_DIR/f'kin_{person}.mat')['kin'],dtype=np.float32,order='C')
        starts=loadmat(REDUCED_DATA_DIR/f'session_start_{person}.mat')['session_start'].ravel().astype(int)-1
        ends=np.r_[starts[1:],len(y)];assert len(starts)==9 and starts[0]==0 and eeg.shape==(len(y),32)
        es,ks=StandardScaler(),StandardScaler()
        for r in range(6):es.partial_fit(eeg[starts[r]:ends[r]]);ks.partial_fit(y[starts[r]:ends[r]])
        for name,sc in [('eeg',es),('target',ks)]:
            np.testing.assert_array_equal(sc.mean_,e['scaling'][name+'_mean']);np.testing.assert_array_equal(sc.scale_,e['scaling'][name+'_scale'])
        assert e['scaling']['samples_seen']==int(es.n_samples_seen_)==sum(ends[:6]-starts[:6])
        eeg=es.transform(eeg,copy=False);y=ks.transform(y,copy=False)
        splits={}
        with np.load(directory/'split_indices.npz') as z:
            for name,runs in d['splits'].items():
                ts=[];rs=[]
                for r in runs:
                    lo,hi=starts[r-1],ends[r-1]
                    for t in range(int(lo)+128,int(hi),64):
                        if t-500-127>=lo:ts.append(t);rs.append(r)
                ts,rs=np.asarray(ts),np.asarray(rs)
                np.testing.assert_array_equal(ts,z[name+'_targets']);np.testing.assert_array_equal(rs,z[name+'_series_ids'])
                assert e['split_shapes'][name]==[len(ts),128,32]
                splits[name]=(ts,rs)
        def features(t,g):return np.asarray([eeg[i-g-127:i-g+1] for i in t],dtype=np.float32)
        for g in [1,25,50,100,250,500]:
            ge=e['gaps'][str(g)];t,_=splits['validation'];x=features(t,g);yv=y[t]
            candidates=ge['ridge']['candidates'];assert [c['alpha'] for c in candidates]==[1.,10.,100.,1000.,10000.]
            for c in candidates:
                assert sha(directory/c['weights'])==c['weights_sha256']
                with np.load(directory/c['weights']) as z:
                    pred=x.reshape(len(x),-1)@z['coef'].T+z['intercept']
                    np.testing.assert_allclose(pred,z['validation_pred'],rtol=2e-5,atol=2e-5)
                    check(yv,z['validation_pred'],c['validation'])
                assert all(0<i<=200 for i in c['n_iter'])
                counts['ridge_axis_fits_at_iteration_cap']+=sum(i==200 for i in c['n_iter'])
                counts['ridge_validation_weight_replays']+=1;counts['validation_metric_sets_checked']+=1
            best=min(candidates,key=lambda c:(c['validation']['mse_mean'],c['alpha']))
            assert best['alpha']==ge['ridge']['selected_alpha'] and best['validation']==ge['ridge']['validation'] and best['weights']==ge['ridge']['weights']
            assert set(ge['cnn_elu'])=={'42','43','44'}
            for seed in [42,43,44]:
                c=ge['cnn_elu'][str(seed)];assert c['initial_sha256']==initial[str(seed)] and c['epochs']==15 and c['parameters']==3235
                assert all(len(v)==15 for v in c['history'].values()) and sha(directory/c['weights'])==c['weights_sha256']
                with np.load(directory/f'cnn_elu_gap{g}_seed{seed}_validation.npz') as z:saved=z['y_pred']
                check(yv,saved,c['validation']);tf.keras.backend.clear_session()
                model=tf.keras.models.load_model(directory/c['weights'],compile=False)
                with tf.device('/GPU:0'):pred=model.predict(x[...,None],batch_size=64,verbose=0)
                np.testing.assert_allclose(pred,saved,rtol=1e-3,atol=5e-4)
                counts['validation_metric_sets_checked']+=1;counts['cnn_validation_weight_replays']+=1;del model
            del x;gc.collect()
        ridge=min([1,25,50,100,250,500],key=lambda g:(e['gaps'][str(g)]['ridge']['validation']['mse_mean'],g))
        cnn=min([1,25,50,100,250,500],key=lambda g:(np.mean([e['gaps'][str(g)]['cnn_elu'][str(s)]['validation']['mse_mean'] for s in [42,43,44]]),g))
        assert d['selection'][person]['ridge']['gap_samples']==ridge and d['selection'][person]['cnn_elu']['gap_samples']==cnn
        permitted={f'ridge_gap{g}' for g in {1,ridge}}|{f'cnn_elu_gap{g}_seed{s}' for g in {1,cnn} for s in [42,43,44]}
        assert set(e['test_evaluations'])==permitted=={p.name.replace('_predictions.npz','') for p in directory.glob('*_predictions.npz')}
        t,ids=splits['test'];yt=y[t]
        for key,c in e['test_evaluations'].items():
            assert c['evaluated_at']>=d['selection_frozen_at']
            with np.load(directory/(key+'_predictions.npz')) as z:
                np.testing.assert_array_equal(z['y_true'],yt);np.testing.assert_array_equal(z['series_id'],ids);np.testing.assert_array_equal(z['target_sample_index'],t);saved=z['y_pred']
            check(yt,saved,c['aggregate']);counts['test_metric_sets_checked']+=1
            for r in [8,9]:mask=ids==r;check(yt[mask],saved[mask],c['by_series'][str(r)]);counts['test_metric_sets_checked']+=1
            g=c['gap_ms']//2;x=features(t,g)
            if c['model']=='ridge':
                with np.load(directory/e['gaps'][str(g)]['ridge']['weights']) as z:pred=x.reshape(len(x),-1)@z['coef'].T+z['intercept']
            else:
                tf.keras.backend.clear_session();model=tf.keras.models.load_model(directory/e['gaps'][str(g)]['cnn_elu'][str(c['seed'])]['weights'],compile=False)
                with tf.device('/GPU:0'):pred=model.predict(x[...,None],batch_size=64,verbose=0)
                del model
            np.testing.assert_allclose(pred,saved,rtol=1e-3 if c['model']=='cnn_elu' else 2e-5,atol=5e-4 if c['model']=='cnn_elu' else 2e-5)
            counts['test_weight_replays']+=1;del x
        counts['participants_checked']+=1;del eeg,y;gc.collect()
    report={'experiment_id':'EXP-042','status':'passed','checked_at':datetime.now(timezone.utc).isoformat(),**counts,'all_six_gaps_both_models_checked':True,'train_only_scalers_recomputed':True,'global_validation_selection_recomputed':True,'permitted_test_files_only':True,'models_trained_by_verifier':0,'verifier_sha256':sha(__file__)}
    (out/'verifier_source.py').write_bytes(Path(__file__).read_bytes())
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
