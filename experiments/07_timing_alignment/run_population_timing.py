"""EXP-042: six gaps, two decoders, P1-P12, validation-only population timing."""
import os
for k,v in {'TF_CPP_MIN_LOG_LEVEL':'2','TF_FORCE_GPU_ALLOW_GROWTH':'true','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','MPLBACKEND':'Agg'}.items():
    os.environ.setdefault(k,v)
import argparse, gc, hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/eeg_kinematics/paths.py').is_file())
sys.path.insert(0,str(ROOT/'src'))
from eeg_kinematics.paths import REDUCED_DATA_DIR, RAW_DATA_DIR
from eeg_kinematics.timing_alignment import matched_targets, make_windows
from run_timing_alignment import build_cnn, metrics
import numpy as np
import scipy, sklearn
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import tensorflow as tf

GAPS=(1,25,50,100,250,500)
SEEDS=(42,43,44)
ALPHAS=(1.,10.,100.,1000.,10000.)
SPLITS={'train':tuple(range(1,7)),'validation':(7,),'test':(8,9)}
PROTOCOL=ROOT/'documentation/methods/POPULATION_TIMING_ALIGNMENT.md'
AUDIT=ROOT/'outputs/condition-audit-20261014T115618Z/results.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def now():return datetime.now(timezone.utc).isoformat()

def save_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)

def dataset(person):
    eeg=np.asarray(loadmat(REDUCED_DATA_DIR/f'eeg_{person}.mat')['eeg'].T,dtype=np.float32,order='C')
    y=np.asarray(loadmat(REDUCED_DATA_DIR/f'kin_{person}.mat')['kin'],dtype=np.float32,order='C')
    starts=loadmat(REDUCED_DATA_DIR/f'session_start_{person}.mat')['session_start'].ravel().astype(np.int64)-1
    assert eeg.shape==(len(y),32) and y.shape[1]==3 and len(starts)==9
    assert starts[0]==0 and np.all(np.diff(starts)>0) and starts[-1]<len(y)
    assert np.isfinite(eeg).all() and np.isfinite(y).all()
    ends=np.r_[starts[1:],len(y)]
    bounds={i+1:(int(lo),int(hi)) for i,(lo,hi) in enumerate(zip(starts,ends))}
    es,ys=StandardScaler(),StandardScaler()
    for r in SPLITS['train']:
        lo,hi=bounds[r];es.partial_fit(eeg[lo:hi]);ys.partial_fit(y[lo:hi])
    meta={'fit_series':list(SPLITS['train']),'samples_seen':int(es.n_samples_seen_),'eeg_mean':es.mean_.tolist(),'eeg_scale':es.scale_.tolist(),'target_mean':ys.mean_.tolist(),'target_scale':ys.scale_.tolist()}
    eeg=es.transform(eeg,copy=False);y=ys.transform(y,copy=False)
    splits={k:matched_targets(bounds,runs,gaps=GAPS) for k,runs in SPLITS.items()}
    for t,ids in splits.values():
        assert len(t)>0
        for r in np.unique(ids):
            lo,hi=bounds[int(r)];mask=ids==r
            assert np.all(t[mask]-627>=lo) and np.all(t[mask]<hi)
    return eeg,y,bounds,splits,meta

def scored(y,p,ids):
    return {'aggregate':metrics(y,p),'by_series':{str(r):metrics(y[ids==r],p[ids==r]) for r in (8,9)}}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();out=a.output_dir.resolve()
    assert out.parent==(ROOT/'outputs').resolve()
    sources={'runner':sha(__file__),'protocol':sha(PROTOCOL),'pilot_runner':sha(Path(__file__).with_name('run_timing_alignment.py')),'timing_helper':sha(ROOT/'src/eeg_kinematics/timing_alignment.py'),'condition_audit':sha(AUDIT)}
    if a.resume:
        d=json.loads((out/'results.json').read_text());assert d['status']!='complete' and d['source_sha256']==sources
    else:
        out.mkdir(exist_ok=False)
        d={'experiment_id':'EXP-042','status':'running','phase':'validation_fitting','created_at':now(),'participants':[f'P{i}' for i in range(1,13)],'gap_samples':list(GAPS),'gap_ms':[g*2 for g in GAPS],'cnn_seeds':list(SEEDS),'window_samples':128,'stride_samples':64,'sampling_hz':500,'splits':{k:list(v) for k,v in SPLITS.items()},'source_sha256':sources,'participant_results':{},'selection':{},'limits':['Exploratory within-participant replication across the 12 dataset participants, not unseen-participant or population-trained decoding.','Historical P1 tests and previous transfer results informed planning.','One validation series; test series are weight series at surface code 3.','Partly undocumented upstream reduced-data preprocessing; index causality is insufficient for online claims.','CNN seeds are averaged before participant aggregation. No ensemble or independent-window inference.','Ridge retains bounded LSQR iterations from the pilot.']}
        for source,target in [(Path(__file__),'runner_source.py'),(PROTOCOL,'declared_protocol.md'),(Path(__file__).with_name('run_timing_alignment.py'),'pilot_runner_source.py'),(ROOT/'src/eeg_kinematics/timing_alignment.py','timing_helper_source.py')]:
            (out/target).write_bytes(source.read_bytes())
    def save():save_json(out/'results.json',d)
    save();print('OUTPUT',str(out),flush=True)
    audit=json.loads(AUDIT.read_text());assert audit['status']=='complete' and audit['source_files_unchanged']
    d['condition_audit']={}
    for person in d['participants']:
        original=RAW_DATA_DIR/person/f'{person}_AllLifts.mat'
        assert sha(original)==audit['source_sha256'][f'../{person}/{person}_AllLifts.mat']
        rows=audit['participants'][person]
        assert all(rows['runs'][str(r)]['series_type']=='weight' and rows['runs'][str(r)]['surfaces']==[3] for r in (7,8,9))
        assert not any(rows['heldout_conditions_absent_from_v2_train'].values())
        d['condition_audit'][person]={'metadata_sha256':sha(original),'test_series_type':'weight','test_surface_code':3,'split_trial_counts':rows['split_trial_counts'],'heldout_conditions_absent_from_training':False}
    gpus=tf.config.list_physical_devices('GPU');assert gpus,'CNN requires GPU'
    for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
    with tf.device('/GPU:0'):probe=tf.matmul(tf.ones((32,32)),tf.ones((32,32)))
    assert 'GPU:0' in probe.device;probe.numpy()
    d['gpu_probe_device']=probe.device;d['gpu_details']=tf.config.experimental.get_device_details(gpus[0]);d['versions']={'python':sys.version.split()[0],'tensorflow':tf.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}
    save()
    # Fit validation candidates for everyone before any new test evaluation.
    for person in d['participants']:
        entry=d['participant_results'].setdefault(person,{'gaps':{},'test_evaluations':{},'cnn_initial_hashes':{}})
        directory=out/person;directory.mkdir(exist_ok=True)
        if len(entry['gaps'])==6 and all('ridge' in z and len(z.get('cnn_elu',{}))==3 for z in entry['gaps'].values()):
            for z in entry['gaps'].values():
                for c in z['ridge']['candidates']+list(z['cnn_elu'].values()):assert sha(directory/c['weights'])==c['weights_sha256']
            print('VALIDATION_ALREADY_COMPLETE',person,flush=True);continue
        eeg,y,bounds,splits,scaling=dataset(person)
        input_hashes={name:sha(REDUCED_DATA_DIR/name) for name in (f'eeg_{person}.mat',f'kin_{person}.mat',f'session_start_{person}.mat')}
        if 'input_sha256' in entry:assert input_hashes==entry['input_sha256'] and scaling==entry['scaling']
        entry.update({'input_sha256':input_hashes,'scaling':scaling,'series_bounds':{str(k):list(v) for k,v in bounds.items()},'split_shapes':{k:[len(t),128,32] for k,(t,ids) in splits.items()}})
        np.savez_compressed(directory/'split_indices.npz',**{k+'_'+label:arr for k,(t,ids) in splits.items() for label,arr in [('targets',t),('series_ids',ids)]})
        save();print('PARTICIPANT',person,'COMMON_TARGETS',entry['split_shapes'],flush=True)
        for gap in GAPS:
            ge=entry['gaps'].setdefault(str(gap),{'gap_ms':gap*2,'cnn_elu':{}})
            if 'ridge' in ge and len(ge['cnn_elu'])==3:continue
            train=make_windows(eeg,*splits['train'],bounds,gap);val=make_windows(eeg,*splits['validation'],bounds,gap)
            yt=y[splits['train'][0]];yv=y[splits['validation'][0]]
            if 'ridge' not in ge:
                candidates=[]
                for alpha in ALPHAS:
                    begin=time.perf_counter()
                    model=Ridge(alpha=alpha,solver='lsqr',tol=.001,max_iter=200).fit(train.reshape(len(train),-1),yt)
                    pred=model.predict(val.reshape(len(val),-1));name=f'ridge_gap{gap}_alpha{alpha:g}.npz'
                    np.savez_compressed(directory/name,coef=model.coef_,intercept=model.intercept_,validation_pred=pred)
                    c={'alpha':alpha,'validation':metrics(yv,pred),'weights':name,'weights_sha256':sha(directory/name),'n_iter':model.n_iter_.tolist(),'fit_seconds':time.perf_counter()-begin}
                    candidates.append(c)
                    print('RIDGE',person,gap*2,alpha,'val_mse',c['validation']['mse_mean'],'n_iter',c['n_iter'],flush=True)
                    del model
                best=min(candidates,key=lambda c:(c['validation']['mse_mean'],c['alpha']))
                ge['ridge']={'selected_alpha':best['alpha'],'validation':best['validation'],'weights':best['weights'],'candidates':candidates};save()
            for seed in SEEDS:
                if str(seed) in ge['cnn_elu']:
                    assert sha(directory/ge['cnn_elu'][str(seed)]['weights'])==ge['cnn_elu'][str(seed)]['weights_sha256'];continue
                tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
                print('CNN_START',person,'gap_ms',gap*2,'seed',seed,flush=True)
                with tf.device('/GPU:0'):
                    model=build_cnn();init=hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
                    if str(seed) in entry['cnn_initial_hashes']:assert entry['cnn_initial_hashes'][str(seed)]==init
                    else:entry['cnn_initial_hashes'][str(seed)]=init
                    begin=time.perf_counter()
                    history=model.fit(train[...,None],yt,validation_data=(val[...,None],yv),epochs=15,batch_size=64,verbose=0)
                    elapsed=time.perf_counter()-begin
                    pred=model.predict(val[...,None],batch_size=64,verbose=0)
                    name=f'cnn_elu_gap{gap}_seed{seed}.keras';model.save(directory/name)
                np.savez_compressed(directory/f'cnn_elu_gap{gap}_seed{seed}_validation.npz',y_pred=pred)
                ge['cnn_elu'][str(seed)]={'validation':metrics(yv,pred),'weights':name,'weights_sha256':sha(directory/name),'initial_sha256':init,'parameters':3235,'epochs':15,'fit_seconds':elapsed,'history':{k:[float(v) for v in a] for k,a in history.history.items()}}
                save();print('CNN_DONE',person,gap*2,seed,'val_mse',ge['cnn_elu'][str(seed)]['validation']['mse_mean'],'seconds',elapsed,flush=True)
                del model,history,pred;gc.collect()
            del train,val,yt,yv;gc.collect()
        del eeg,y;gc.collect()
    selection={}
    for person in d['participants']:
        gaps=d['participant_results'][person]['gaps']
        r=min(GAPS,key=lambda g:(gaps[str(g)]['ridge']['validation']['mse_mean'],g))
        c=min(GAPS,key=lambda g:(np.mean([gaps[str(g)]['cnn_elu'][str(s)]['validation']['mse_mean'] for s in SEEDS]),g))
        selection[person]={'ridge':{'gap_samples':r,'gap_ms':r*2,'alpha':gaps[str(r)]['ridge']['selected_alpha']},'cnn_elu':{'gap_samples':c,'gap_ms':c*2,'validation_mse_seed_mean':float(np.mean([gaps[str(c)]['cnn_elu'][str(s)]['validation']['mse_mean'] for s in SEEDS]))}}
    if (out/'selection.json').exists():
        frozen=json.loads((out/'selection.json').read_text());assert frozen['participants']==selection
    else:
        frozen={'frozen_at':now(),'participants':selection,'criterion':'validation-only mean standardized XYZ MSE; CNN mean of seeds42/43/44; shorter-gap tie break; all participants before test'}
        save_json(out/'selection.json',frozen)
    d['selection']=selection;d['selection_frozen_at']=frozen['frozen_at'];d['phase']='testing';save()
    print('ALL_POPULATION_SELECTIONS_FROZEN',flush=True)
    for person in d['participants']:
        entry=d['participant_results'][person];directory=out/person
        eeg,y,bounds,splits,scaling=dataset(person);assert scaling==entry['scaling']
        assert all(sha(REDUCED_DATA_DIR/name)==h for name,h in entry['input_sha256'].items())
        t,ids=splits['test'];yt=y[t]
        for kind in ['ridge','cnn_elu']:
            for gap in sorted({1,selection[person][kind]['gap_samples']}):
                x=make_windows(eeg,t,ids,bounds,gap)
                for seed in ([None] if kind=='ridge' else SEEDS):
                    key=f'{kind}_gap{gap}'+('' if seed is None else f'_seed{seed}')
                    if key in entry['test_evaluations']:
                        assert (directory/f'{key}_predictions.npz').is_file();continue
                    if seed is None:
                        with np.load(directory/entry['gaps'][str(gap)][kind]['weights']) as z:pred=x.reshape(len(x),-1)@z['coef'].T+z['intercept']
                    else:
                        tf.keras.backend.clear_session()
                        model=tf.keras.models.load_model(directory/entry['gaps'][str(gap)][kind][str(seed)]['weights'],compile=False)
                        with tf.device('/GPU:0'):pred=model.predict(x[...,None],batch_size=64,verbose=0)
                        del model
                    np.savez_compressed(directory/f'{key}_predictions.npz',y_true=yt,y_pred=pred,target_sample_index=t,series_id=ids)
                    entry['test_evaluations'][key]={'evaluated_at':now(),'model':kind,'gap_ms':gap*2,'seed':seed,**scored(yt,pred,ids)};save()
                    print('TEST_DONE',person,key,flush=True)
                del x;gc.collect()
        del eeg,y;gc.collect()
    assert all(sha(p)==sources[k] for k,p in [('runner',Path(__file__)),('protocol',PROTOCOL),('pilot_runner',Path(__file__).with_name('run_timing_alignment.py')),('timing_helper',ROOT/'src/eeg_kinematics/timing_alignment.py'),('condition_audit',AUDIT)])
    d['source_files_unchanged']=True;d['status']='complete';d['phase']='complete';d['completed_at']=now();save()
    print('ALL_FITS_AND_TESTS_COMPLETE',str(out),flush=True)

if __name__=='__main__':main()
