"""Independent numerical verification of EXP-036. No model fitting or inference."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,csv,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/eeg_kinematics/paths.py').is_file());sys.path.insert(0,str(ROOT/'src'))
from eeg_kinematics.paths import REDUCED_DATA_DIR
import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_coeffs,savgol_filter
from scipy.stats import rankdata

STAGES=['unassigned/intertrial','preparation','reach','preload','loading','airborne_before_lowering_cue','after_lowering_cue','release','return']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open()))
def weights(k):
    w=np.array([1/np.count_nonzero(k==v) for v in k]);return w/w.mean()
def quadratic(x):
    return np.concatenate((x,np.stack([x[:,i]*x[:,j] for i in range(x.shape[1]) for j in range(i,x.shape[1])],axis=1)),axis=1)
def checkscore(y,p,k,expected):
    u=np.unique(k);mse=np.mean([np.mean((p[k==v]-y[k==v])**2) for v in u]);mae=np.mean([np.mean(np.abs(p[k==v]-y[k==v])) for v in u]);w=weights(k);mean=np.sum(y*w)/sum(w);var=np.sum(w*(y-mean)**2)/sum(w)
    assert len(y)==expected['n'] and len(u)==expected['trials']
    np.testing.assert_allclose([mse,mae,1-mse/var,np.mean(p<0)],[expected['mse'],expected['mae'],expected['r2_error_target'],expected['negative_prediction_fraction']],rtol=1e-10,atol=1e-12)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output_dir',type=Path);args=ap.parse_args();out=args.output_dir.resolve();assert out.parent==(ROOT/'outputs').resolve()
    d=json.loads((out/'results.json').read_text());z=np.load(out/'analysis_data.npz');fits=np.load(out/'fitted_explainers.npz');assert d['experiment_id']=='EXP-036' and d['status']=='complete' and not d['decoder_training_or_inference']
    for name,h in d['snapshot_hashes'].items():assert sha(out/name)==h
    for name,h in d['source_hashes'].items():assert sha(ROOT/name)==h
    pos=loadmat(REDUCED_DATA_DIR/'kin_P1.mat')['kin'];starts=loadmat(REDUCED_DATA_DIR/'session_start_P1.mat')['session_start'].ravel().astype(int)-1;ends=np.r_[starts[1:],len(pos)]
    idx=z['target_index'];run=z['series'];trial=z['trial'];stage=z['stage'];key=z['trial_key'];primary=z['primary'];eligible=z['eligible'];oof=z['oof'];models=d['models'];specs=list(d['feature_specs'])
    np.testing.assert_array_equal(z['position'],pos[idx]);np.testing.assert_array_equal(key,run*100+trial)
    events=json.loads((ROOT/'outputs/saved-decoder-analysis-p1-20261013T172654Z/trial_events.json').read_text())
    for r in [8,9]:
        mask=run==r;local=idx[mask]-starts[r-1];length=ends[r-1]-starts[r-1];expected_stage=np.zeros(length,int);expected_trial=np.zeros(length,int);expected_pure=np.zeros(length,bool)
        for t in events[str(r)]:
            b=np.rint(np.array(t['events'])*500).astype(int)
            for j in range(8):expected_stage[b[j]:b[j+1]]=j+1;expected_trial[b[j]:b[j+1]]=t['trial']
        np.testing.assert_array_equal(stage[mask],expected_stage[local]);np.testing.assert_array_equal(trial[mask],expected_trial[local]);np.testing.assert_array_equal(eligible[mask],(local>=40)&(local<length-40))
    np.testing.assert_array_equal(primary,eligible&(trial>0));assert primary.sum()==d['n_primary_targets'] and len(np.unique(key[primary]))==68
    meta=json.loads((ROOT/'outputs/six-model-comparison-p1-gpu-20260930T095506Z/results.json').read_text());sc=meta['scaling'];sourcechecks=0
    for im,m in enumerate(models):
        q=np.load(ROOT/f'outputs/six-model-comparison-p1-gpu-20260930T095506Z/{m}_predictions.npz');order=np.lexsort((q['target_sample_index'],q['run_id']));np.testing.assert_array_equal(idx,q['target_sample_index'][order]);np.testing.assert_array_equal(run,q['run_id'][order])
        np.testing.assert_allclose(z['observed'][im],q['y_true'][order]*sc['target_scale']+sc['target_mean'],atol=1e-12,rtol=1e-12);np.testing.assert_allclose(z['predicted'][im],q['y_pred'][order]*sc['target_scale']+sc['target_mean'],atol=1e-12,rtol=1e-12);sourcechecks+=1
    np.testing.assert_array_equal(z['error'],np.sqrt(np.sum((z['predicted']-z['observed'])**2,axis=2)))
    derivative_checks=fitchecks=metricchecks=trialchecks=0
    correlation_checks=stage_trial_checks=bin_checks=0
    all_correlations=rows(out/'within_stage_correlations.csv');all_stages=rows(out/'stage_trial_dynamics.csv');all_bins=rows(out/'speed_bins.csv')
    for wi,window in enumerate(d['smoothing_windows_samples']):
        half=window//2;offset=np.arange(-half,half+1)
        for r in [8,9]:
            mask=(run==r)&eligible;points=idx[mask];observed_windows=pos[points[:,None]+offset]
            for derivative in [1,2,3]:
                coef=savgol_coeffs(window,3,deriv=derivative,delta=.002,use='dot');expected=np.einsum('nwk,w->nk',observed_windows,coef)
                np.testing.assert_allclose(z['derivatives'][wi,mask,derivative-1],expected,rtol=1e-7,atol=[1e-9,1e-7,2e-5][derivative-1]);derivative_checks+=int(mask.sum())
        v,a,j=z['derivatives'][wi,:,0],z['derivatives'][wi,:,1],z['derivatives'][wi,:,2]
        group={'P':quadratic(z['position']),'S':(stage[:,None]==np.arange(1,9)).astype(float),'D':quadratic(np.concatenate([v,a,np.sqrt((v*v).sum(1))[:,None],np.sqrt((a*a).sum(1))[:,None]],axis=1)),'J':quadratic(np.concatenate([j,np.sqrt((j*j).sum(1))[:,None]],axis=1))}
        for im,m in enumerate(models):
            y=z['error'][im]
            for si,s in enumerate(specs):
                chars=d['feature_specs'][s];X=np.concatenate([group[c] for c in chars],axis=1) if chars else None
                for trainrun,testrun in [(8,9),(9,8)]:
                    aa=primary&(run==trainrun);bb=primary&(run==testrun);prefix=f'w{window}_{m}_{s}_train{trainrun}'
                    np.testing.assert_array_equal(fits[prefix+'_train_indices'],np.flatnonzero(aa));np.testing.assert_array_equal(fits[prefix+'_eval_indices'],np.flatnonzero(bb));assert not set(key[aa])&set(key[bb])
                    w=weights(key[aa]);intercept=float(fits[prefix+'_intercept'])
                    if X is None:
                        np.testing.assert_allclose(intercept,np.average(y[aa],weights=w));expected=np.full(bb.sum(),intercept)
                    else:
                        mean=np.sum(X[aa]*w[:,None],axis=0)/sum(w);var=np.sum((X[aa]-mean)**2*w[:,None],axis=0)/sum(w);scale=np.sqrt(var);scale[scale<1e-14]=1
                        np.testing.assert_allclose(mean,fits[prefix+'_mean'],rtol=1e-10,atol=1e-11);np.testing.assert_allclose(scale,fits[prefix+'_scale'],rtol=1e-10,atol=1e-11)
                        coef=fits[prefix+'_coef'];T=(X[aa]-mean)/scale;expected=(X[bb]-mean)/scale@coef+intercept
                        residual=T@coef+intercept-y[aa]
                        np.testing.assert_allclose(T.T@(w*residual)+10*coef,0,atol=2e-6);np.testing.assert_allclose(np.sum(w*residual),0,atol=2e-7)
                    np.testing.assert_allclose(expected,oof[wi,im,si,bb],rtol=1e-9,atol=1e-10);fitchecks+=1
                    target=next(f for f in d['scores'][str(window)][m][s]['folds'] if f['eval_series']==testrun);checkscore(y[bb],expected,key[bb],target);metricchecks+=1
                checkscore(y[primary],oof[wi,im,si,primary],key[primary],d['scores'][str(window)][m][s]['combined']);metricchecks+=1
        speed=np.linalg.norm(v,axis=1);acc=np.linalg.norm(a,axis=1);jerk=np.linalg.norm(j,axis=1)
        fields={'speed':speed,'velocity_x':v[:,0],'velocity_y':v[:,1],'velocity_z':v[:,2],'acceleration':acc,'jerk':jerk}
        for r in [r for r in all_correlations if int(r['window_samples'])==window]:
            im=models.index(r['decoder']);q=primary&(stage==STAGES.index(r['stage']));x=fields[r['quantity']][q];y=z['error'][im,q]
            assert q.sum()==int(r['n']) and len(np.unique(key[q]))==int(r['trials'])
            if r['spearman_rho']:
                np.testing.assert_allclose(np.corrcoef(rankdata(x),rankdata(y))[0,1],float(r['spearman_rho']),atol=1e-12)
            correlation_checks+=1
        for r in [r for r in all_stages if int(r['window_samples'])==window]:
            im=models.index(r['decoder']);q=primary&(key==int(r['series'])*100+int(r['trial']))&(stage==STAGES.index(r['stage']))
            assert q.sum()==int(r['n']) and z['pure_window'][q].sum()==int(r['pure_window_n'])
            np.testing.assert_allclose([z['error'][im,q].mean(),speed[q].mean(),acc[q].mean(),jerk[q].mean()],[float(r[c]) for c in ['mean_error','mean_speed','mean_acceleration','mean_jerk']],rtol=1e-10,atol=1e-12);stage_trial_checks+=1
        train_speeds=[]
        for r in range(1,7):
            lo,hi=starts[r-1],ends[r-1];vv=savgol_filter(pos[lo:hi].astype(float),window,3,deriv=1,delta=.002,axis=0)
            train_speeds.extend(np.linalg.norm(vv[40:-40],axis=1))
        edges=np.r_[0,np.unique(np.quantile(train_speeds,[.2,.4,.6,.8,.9,.95,.99])),np.inf]
        for r in [r for r in all_bins if int(r['window_samples'])==window]:
            im=models.index(r['decoder']);bi=int(r['speed_bin']);q=primary&(speed>=edges[bi])&(speed<edges[bi+1]);q&=True if r['stage']=='all_trial_stages' else stage==STAGES.index(r['stage'])
            means=[z['error'][im,q&(key==k)].mean() for k in np.unique(key[q])]
            assert q.sum()==int(r['n']) and len(means)==int(r['trials'])
            np.testing.assert_allclose([np.median(speed[q]),np.mean(means),np.quantile(means,.25),np.quantile(means,.75)],[float(r[c]) for c in ['median_speed','trial_mean_error','trial_q25','trial_q75']],atol=1e-12,rtol=1e-10);bin_checks+=1
        print('VERIFIED_WINDOW',window,flush=True)
    assert np.isnan(oof[:,:,:,~primary]).all()
    for r in rows(out/'explanatory_trials.csv'):
        wi=d['smoothing_windows_samples'].index(int(r['window_samples']));mi=models.index(r['decoder']);si=specs.index(r['features']);q=primary&(key==int(r['series'])*100+int(r['trial']));res=oof[wi,mi,si,q]-z['error'][mi,q]
        assert q.sum()==int(r['n']);np.testing.assert_allclose([np.mean(res**2),np.mean(abs(res))],[float(r['mse']),float(r['mae'])],rtol=1e-10,atol=1e-12);trialchecks+=1
    primary_rows=rows(out/'primary_paired_trials.csv');delta=[];trialkeys=np.unique(key[primary]);wi=d['smoothing_windows_samples'].index(41);mi=models.index('cnn_bilstm');f=specs.index('combined');b=specs.index('position_stage')
    for k,r in zip(trialkeys,primary_rows):
        q=primary&(key==k);yy=z['error'][mi,q];value=np.mean((oof[wi,mi,f,q]-yy)**2)-np.mean((oof[wi,mi,b,q]-yy)**2);np.testing.assert_allclose(value,float(r['combined_minus_position_stage_mse']),atol=1e-12);delta.append(value)
    delta=np.array(delta);np.testing.assert_allclose(delta.mean(),d['primary_contrast']['mean_paired_trial_delta'],atol=1e-12);assert sum(delta<0)==d['primary_contrast']['trials_improved']
    for it in d['primary_contrast']['intervals']:
        rng=np.random.default_rng(it['seed']);distribution=[];block=it['block_trials']
        for _ in range(it['replicates']):
            samples=[]
            for r in [8,9]:
                a=delta[trialkeys//100==r];start=rng.integers(0,len(a),size=int(np.ceil(len(a)/block)));ii=np.array([(v+j)%len(a) for v in start for j in range(block)])[:len(a)];samples.extend(a[ii])
            distribution.append(np.mean(samples))
        np.testing.assert_allclose(np.quantile(distribution,[.025,.975]),[it['lower_95'],it['upper_95']],atol=1e-12)
    regionchecks=0;edges=np.linspace(0,1,51);speed=np.linalg.norm(z['derivatives'][wi,:,0],axis=1)
    for axis in range(3):
        anchors=np.argsort(-np.histogram(pos[:starts[6],axis],edges)[0],kind='stable')[:5];assert anchors.tolist()==d['anchors_by_axis'][axis]
        for im in range(6):
            yy=z['observed'][im,:,axis];pp=z['predicted'][im,:,axis];ob=np.clip(np.digitize(yy,edges)-1,0,49);pb=np.clip(np.digitize(pp,edges)-1,0,49)
            oa=np.isin(ob,anchors)&(yy>=0)&(yy<=1);pa=np.isin(pb,anchors)&(pp>=0)&(pp<=1);error=np.abs(pp-yy);high=error>=.10
            code=np.full(len(idx),5);code[error<=.05]=0;code[high&oa&~pa]=1;code[high&~oa&pa]=2;code[high&oa&pa]=3;code[high&~oa&~pa]=4
            np.testing.assert_array_equal(code,z['region_codes'][im,axis])
    for r in rows(out/'scatter_regions.csv'):
        im=models.index(r['decoder']);axis='XYZ'.index(r['axis']);labels=['near_diagonal','observed_anchor_only_high_error','predicted_anchor_only_high_error','both_anchors_high_error','neither_anchor_high_error','intermediate_error'];q=eligible&(z['region_codes'][im,axis]==labels.index(r['region']))
        assert q.sum()==int(r['n']) and (q&primary).sum()==int(r['trial_samples']);assert len(np.unique(key[q&primary]))==int(r['trials'])
        if q.any():np.testing.assert_allclose([np.median(speed[q]),np.mean(speed[q]<=.02)],[float(r['median_speed']),float(r['low_speed_fraction'])],atol=1e-12)
        regionchecks+=1
    (out/'verification_source.py').write_bytes(Path(__file__).read_bytes())
    result={'experiment_id':'EXP-036','status':'passed','checked_at':datetime.now(timezone.utc).isoformat(),'source_models_checked':sourcechecks,'saved_prediction_targets_per_model':len(idx),'trial_event_assignments_checked':68,'derivative_values_checked':derivative_checks*3,'coefficient_replays_and_ridge_stationarity_checks':fitchecks,'weighted_metric_sets_checked':metricchecks,'trial_metric_rows_checked':trialchecks,'within_stage_correlations_checked':correlation_checks,'stage_trial_rows_checked':stage_trial_checks,'speed_bin_summaries_checked':bin_checks,'scatter_region_summaries_checked':regionchecks,'anchor_and_region_definitions_recomputed':True,'bootstrap_intervals_recomputed':3,'source_and_protocol_hashes_match':True,'verifier_sha256':sha(out/'verification_source.py'),'new_models_fitted_by_verifier':0}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
