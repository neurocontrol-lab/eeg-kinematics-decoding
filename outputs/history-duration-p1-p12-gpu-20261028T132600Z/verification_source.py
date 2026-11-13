"""Independent saved-evidence checks for EXP-018; no model fitting."""
import os
for key, value in {'TF_CPP_MIN_LOG_LEVEL':'2','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','TF_FORCE_GPU_ALLOW_GROWTH':'true'}.items():
    os.environ.setdefault(key, value)
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/eeg_kinematics/paths.py').is_file())
sys.path.insert(0, str(ROOT/'src'))
from eeg_kinematics.paths import REDUCED_DATA_DIR


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def metric_check(y, pred, recorded):
    y, pred = y.astype(float), pred.astype(float)
    assert len(y) == recorded['n'] and y.shape == pred.shape and np.isfinite(pred).all()
    mse = mean_squared_error(y, pred, multioutput='raw_values')
    np.testing.assert_allclose(mse, recorded['mse'], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mse.mean(), recorded['mse_mean'], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.sqrt(mse), recorded['rmse'], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mean_absolute_error(y,pred,multioutput='raw_values'), recorded['mae'], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(r2_score(y,pred,multioutput='raw_values'), recorded['r2'], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(np.linalg.norm(pred-y,axis=1).mean(), recorded['mean_error_norm'], rtol=1e-10, atol=1e-12)


def independent_windows(eeg, targets, history):
    result = np.empty((len(targets),128,32), np.float32)
    offsets = np.arange(-history,0)
    for start in range(0,len(targets),256):
        t = targets[start:start+256]
        raw = eeg[t[:,None]+offsets[None,:]]
        result[start:start+len(t)] = raw.reshape(len(t),128,history//128,32).mean(axis=2)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output_dir',type=Path);args=ap.parse_args()
    out=args.output_dir.resolve();assert out.parent==(ROOT/'outputs').resolve()
    d=json.loads((out/'results.json').read_text());assert d['status']=='complete' and d['experiment_id']=='EXP-018'
    for key,name in [('runner','experiment_source.py'),('helper','history_source.py'),('protocol','declared_protocol.md'),('reference_model','reference_model_source.py')]:
        assert sha(out/name)==d['source_sha256'][key]
    assert d['source_files_unchanged'] and d['histories_samples']==[128,256,512] and d['input_shape']==[128,32]
    assert d['runs']=={'train':list(range(1,7)),'validation':[7],'test':[8,9]} and d['end_gap_samples']==1 and d['stride_samples']==64
    frozen=json.loads((out/'selection.json').read_text());assert len(frozen['models'])==12 and frozen['frozen_at']==d['selection_frozen_at']
    gpu=tf.config.list_physical_devices('GPU');assert gpu
    for g in gpu:tf.config.experimental.set_memory_growth(g,True)
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
    metric_sets=0;replays=0;windows=0
    for pid in range(1,13):
        label=f'P{pid}';p=d['participants'][label];assert p['fitting_complete']
        for name,h in p['input_sha256'].items():assert sha(REDUCED_DATA_DIR/name)==h
        eeg=np.asarray(loadmat(REDUCED_DATA_DIR/f'eeg_P{pid}.mat')['eeg'].T,dtype=np.float32,order='C')
        kin=np.asarray(loadmat(REDUCED_DATA_DIR/f'kin_P{pid}.mat')['kin'],dtype=np.float32,order='C')
        starts=loadmat(REDUCED_DATA_DIR/f'session_start_P{pid}.mat')['session_start'].ravel().astype(int)-1
        ends=np.r_[starts[1:],len(eeg)];assert len(starts)==9 and starts[0]==0
        es,ks=StandardScaler(),StandardScaler()
        for lo,hi in zip(starts[:6],ends[:6]):es.partial_fit(eeg[lo:hi]);ks.partial_fit(kin[lo:hi])
        for prefix,sc in [('eeg',es),('target',ks)]:
            np.testing.assert_array_equal(sc.mean_,p['scaling'][prefix+'_mean'])
            np.testing.assert_array_equal(sc.scale_,p['scaling'][prefix+'_scale'])
        assert int(es.n_samples_seen_)==p['scaling']['samples_seen'] and p['scaling']['fit_series']==list(range(1,7))
        eeg=es.transform(eeg,copy=False);kin=ks.transform(kin,copy=False)
        splits={}
        with np.load(out/f'{label}_indices.npz') as z:
            for split,runs in d['runs'].items():
                all_t=[];all_r=[]
                for run in runs:
                    lo,hi=int(starts[run-1]),int(ends[run-1]);assert p['bounds'][str(run)]==[lo,hi]
                    t=np.array([i for i in range(lo+128,hi,64) if i-512>=lo],dtype=np.int64)
                    all_t.extend(t);all_r.extend([run]*len(t))
                    for h in (128,256,512):assert np.all(t-h>=lo) and np.all(t<hi)
                t,r=np.array(all_t),np.array(all_r)
                np.testing.assert_array_equal(t,z[split+'_targets']);np.testing.assert_array_equal(r,z[split+'_run_ids'])
                assert len(t)==p['split_counts'][split];splits[split]=(t,r)
        for h in (128,256,512):
            entry=p['histories'][str(h)];tv,rv=splits['validation'];yv=kin[tv]
            xv=independent_windows(eeg,tv,h);windows+=len(xv)
            candidates=entry['ridge']['candidates'];assert [c['alpha'] for c in candidates]==[1.,10.,100.,1000.,10000.]
            for candidate in candidates:
                with np.load(out/candidate['weights']) as z:
                    pred=xv.reshape(len(xv),-1)@z['coef'].T+z['intercept']
                    np.testing.assert_allclose(pred,z['validation_pred'],rtol=1e-5,atol=1e-5)
                    metric_check(yv,z['validation_pred'],candidate['validation']);metric_sets+=1;replays+=1
                assert all(0<i<=200 for i in candidate['n_iter'])
            best=min(candidates,key=lambda c:(c['validation']['mse_mean'],c['alpha']))
            assert best['alpha']==entry['ridge']['alpha'] and best['weights']==entry['ridge']['weights']
            ce=entry['cnn_elu'];assert ce['parameters']==3235 and ce['epochs']==15 and ce['initial_sha256']==d['cnn_initial_sha256']
            assert len(ce['history']['loss'])==15 and len(ce['history']['val_loss'])==15
            tf.keras.backend.clear_session()
            cnn=tf.keras.models.load_model(out/ce['weights'],compile=False);assert cnn.input_shape==(None,128,32,1) and cnn.count_params()==3235
            with np.load(out/f'{label}_cnn_elu_h{h}_validation.npz') as z:
                with tf.device('/GPU:0'):pred=cnn.predict(xv[...,None],batch_size=64,verbose=0)
                np.testing.assert_allclose(pred,z['y_pred'],rtol=1e-4,atol=1e-4)
                metric_check(yv,z['y_pred'],ce['validation']);metric_sets+=1;replays+=1
            del xv
            tt,rt=splits['test'];yt=kin[tt];xt=independent_windows(eeg,tt,h);windows+=len(xt)
            for model in ('ridge','cnn_elu'):
                e=p['test'][f'{model}_{h}']
                with np.load(out/e['predictions']) as z:
                    np.testing.assert_array_equal(z['target_sample_index'],tt);np.testing.assert_array_equal(z['run_id'],rt)
                    np.testing.assert_array_equal(z['y_true'],yt);recorded=z['y_pred'].copy()
                if model=='ridge':
                    with np.load(out/entry['ridge']['weights']) as z:pred=xt.reshape(len(xt),-1)@z['coef'].T+z['intercept']
                else:
                    with tf.device('/GPU:0'):pred=cnn.predict(xt[...,None],batch_size=64,verbose=0)
                np.testing.assert_allclose(pred,recorded,rtol=1e-4,atol=1e-4);replays+=1
                metric_check(yt,recorded,e['aggregate']);metric_sets+=1
                for run in (8,9):metric_check(yt[rt==run],recorded[rt==run],e['by_series'][str(run)]);metric_sets+=1
            del xt,cnn
        for model in ('ridge','cnn_elu'):
            selected=min((128,256,512),key=lambda h:(p['histories'][str(h)][model]['validation']['mse_mean'],h))
            assert selected==p['selection'][model]==frozen['models'][label][model]
        print('VERIFIED_PARTICIPANT',label,flush=True)
        del eeg,kin
    rows=list(csv.DictReader((out/'participant_effects.csv').open()));assert len(rows)==48
    rng=np.random.default_rng(20261028);draws=rng.integers(0,12,size=(20000,12))
    for model in ('ridge','cnn_elu'):
        for ms in (512,1024):
            rel=[];delta=[]
            for pid in range(1,13):
                p=d['participants'][f'P{pid}'];base=p['test'][f'{model}_128']['aggregate']['mse_mean'];longer=p['test'][f'{model}_{ms//2}']['aggregate']['mse_mean']
                rel.append(100*(base-longer)/base);delta.append(longer-base)
                row=next(r for r in rows if r['participant']==f'P{pid}' and r['model']==model and int(r['history_ms'])==ms)
                np.testing.assert_allclose(float(row['relative_mse_reduction_pct']),rel[-1],rtol=1e-12,atol=1e-12)
            rel,delta=np.array(rel),np.array(delta);entry=d['population'][f'{model}_{ms}ms']
            assert entry['participants']==12 and entry['improved']==int((rel>0).sum())
            np.testing.assert_allclose([rel.mean(),np.median(rel),delta.mean()],[entry['mean_relative_mse_reduction_pct'],entry['median_relative_mse_reduction_pct'],entry['mean_mse_difference']],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(np.quantile(rel[draws].mean(axis=1),[.025,.975]),entry['mean_relative_reduction_bootstrap_95'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(np.quantile(delta[draws].mean(axis=1),[.025,.975]),entry['mean_mse_difference_bootstrap_95'],rtol=1e-12,atol=1e-12)
    group_checks=0
    if (out/'group_summary.json').exists():
        group=json.loads((out/'group_summary.json').read_text())
        receipt=json.loads((out/'summary_receipt.json').read_text())
        assert sha(out/'summary_source.py')==receipt['source_sha256'] and sha(out/'results.json')==receipt['results_sha256']
        for name,digest in receipt['artifacts'].items():assert sha(out/name)==digest
        for row in group['fixed_histories']:
            scores=[d['participants'][f'P{i}']['test'][f"{row['model']}_{row['history_ms']//2}"]['aggregate'] for i in range(1,13)]
            np.testing.assert_allclose(row['macro_mse'],np.mean([s['mse_mean'] for s in scores]),rtol=1e-12)
            np.testing.assert_allclose([row['macro_r2_x'],row['macro_r2_y'],row['macro_r2_z']],np.mean([s['r2'] for s in scores],axis=0),rtol=1e-12)
            assert row['participants']==12;group_checks+=1
        for model in ('ridge','cnn_elu'):
            chosen=[frozen['models'][f'P{i}'][model] for i in range(1,13)]
            scores=[d['participants'][f'P{i}']['test'][f'{model}_{h}']['aggregate'] for i,h in enumerate(chosen,1)]
            rec=group['validation_selected_histories'][model]
            assert rec['history_ms_counts']=={str(h*2):chosen.count(h) for h in (128,256,512)}
            np.testing.assert_allclose(rec['macro_mse'],np.mean([s['mse_mean'] for s in scores]),rtol=1e-12);group_checks+=1
    record={'experiment_id':'EXP-018','status':'passed','checked_at':datetime.now(timezone.utc).isoformat(),
            'participants':12,'validation_metric_sets':216,'total_metric_sets':metric_sets,'saved_weight_replays':replays,
            'independently_built_windows':windows,'participant_contrasts':4,'participant_effect_rows':48,
            'train_only_scalers_and_common_targets_recomputed':True,'frozen_validation_selections_checked':True,
            'source_input_and_protocol_hashes_checked':True,'macro_group_checks':group_checks,'models_fitted_by_verifier':0,'verifier_sha256':sha(__file__)}
    assert metric_sets==432 and replays==288
    (out/'verification_source.py').write_bytes(Path(__file__).read_bytes())
    (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
