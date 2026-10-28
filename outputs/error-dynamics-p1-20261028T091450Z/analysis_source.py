"""EXP-036: retrospective dynamics and out-of-series explanation of saved errors."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,csv,hashlib,json,platform,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/eeg_kinematics/paths.py').is_file())
sys.path.insert(0,str(ROOT/'src'))
from eeg_kinematics.paths import REDUCED_DATA_DIR
from eeg_kinematics.error_dynamics import STAGES,MODELS,WINDOWS,SPECS,derivatives,groups,trial_weights,scores,block_interval
import numpy as np
import scipy,sklearn
from scipy.io import loadmat
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

SOURCE=ROOT/'outputs/six-model-comparison-p1-gpu-20260930T095506Z'
PREVIOUS=ROOT/'outputs/saved-decoder-analysis-p1-20261013T172654Z'
PROTOCOL=ROOT/'documentation/methods/ERROR_DYNAMICS.md'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def finishfig(fig,p):fig.savefig(p,dpi=160,bbox_inches='tight');plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path);args=parser.parse_args()
    out=(args.output_dir or ROOT/'outputs'/('error-dynamics-p1-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    assert out.parent==(ROOT/'outputs').resolve();out.mkdir(exist_ok=False)
    snapshots={'analysis_source.py':Path(__file__),'feature_source.py':ROOT/'src/eeg_kinematics/error_dynamics.py','declared_protocol.md':PROTOCOL}
    for dst,src in snapshots.items():(out/dst).write_bytes(src.read_bytes())
    sourcefiles=[SOURCE/'results.json',PREVIOUS/'results.json',PREVIOUS/'verification.json',PREVIOUS/'trial_events.json',PREVIOUS/'aligned_predictions.csv',REDUCED_DATA_DIR/'kin_P1.mat',REDUCED_DATA_DIR/'session_start_P1.mat']+[SOURCE/f'{m}_predictions.npz' for m in MODELS]
    sourcehash={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else '../'+str(p.relative_to(ROOT.parent)):sha(p) for p in sourcefiles}
    prior=json.loads((PREVIOUS/'results.json').read_text());meta=json.loads((SOURCE/'results.json').read_text())
    assert json.loads((PREVIOUS/'verification.json').read_text())['status']=='passed'
    for p,h in prior['source_sha256'].items():assert sha(ROOT/p)==h
    trace=list(csv.DictReader((PREVIOUS/'aligned_predictions.csv').open()))
    lead=[r for r in trace if r['model']=='cnn_bilstm'];n=len(lead);assert n==3642
    idx=np.array([int(r['target_index']) for r in lead]);run=np.array([int(r['run']) for r in lead]);trial=np.array([int(r['trial']) for r in lead]);stage=np.array([STAGES.index(r['stage']) for r in lead]);keys=run*100+trial
    time=np.array([float(r['run_time_seconds']) for r in lead]);pure=np.array([r['pure_window']=='True' for r in lead])
    kin=loadmat(REDUCED_DATA_DIR/'kin_P1.mat')['kin'];starts=loadmat(REDUCED_DATA_DIR/'session_start_P1.mat')['session_start'].ravel().astype(int)-1;ends=np.r_[starts[1:],len(kin)]
    position=kin[idx].astype(float);observed=np.empty((6,n,3));predicted=np.empty_like(observed)
    km=np.array(meta['scaling']['target_mean']);ks=np.array(meta['scaling']['target_scale'])
    for mi,m in enumerate(MODELS):
        rr=[r for r in trace if r['model']==m];z=np.load(SOURCE/f'{m}_predictions.npz');order=np.lexsort((z['target_sample_index'],z['run_id']))
        np.testing.assert_array_equal(idx,z['target_sample_index'][order]);np.testing.assert_array_equal(run,z['run_id'][order])
        yy=z['y_true'][order]*ks+km;pp=z['y_pred'][order]*ks+km
        observed[mi]=[[float(r['y_'+a]) for a in 'xyz'] for r in rr];predicted[mi]=[[float(r['pred_'+a]) for a in 'xyz'] for r in rr]
        np.testing.assert_allclose(observed[mi],yy,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(predicted[mi],pp,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(yy,position,rtol=1e-6,atol=1e-7)
        assert all((int(r['trial']),r['stage'],int(r['target_index']))==(int(trial[i]),STAGES[stage[i]],int(idx[i])) for i,r in enumerate(rr))
    events=json.loads((PREVIOUS/'trial_events.json').read_text());weight=np.zeros(n,int);eligible=np.ones(n,bool)
    for r in [8,9]:
        mask=run==r;lo,hi=starts[r-1],ends[r-1];local=idx[mask]-lo
        np.testing.assert_allclose(time[mask],local/500,atol=1e-12)
        eligible[mask]=(local>=40)&(local<hi-lo-40)
        ph=np.zeros(hi-lo,int);tr=np.zeros_like(ph)
        for t in events[str(r)]:
            b=np.rint(np.array(t['events'])*500).astype(int)
            for j in range(8):ph[b[j]:b[j+1]]=j+1;tr[b[j]:b[j+1]]=t['trial']
            weight[mask&(trial==t['trial'])]=t['weight_code']
        np.testing.assert_array_equal(stage[mask],ph[local]);np.testing.assert_array_equal(trial[mask],tr[local])
    primary=eligible&(trial>0);assert len(np.unique(keys[primary]))==68
    error=np.linalg.norm(predicted-observed,axis=2)
    dyn=np.empty((3,n,3,3));bounds={};bin_edges={}
    for wi,window in enumerate(WINDOWS):
        train=[]
        for r in range(1,10):
            lo,hi=starts[r-1],ends[r-1];d=derivatives(kin[lo:hi].astype(float),window)
            if r<=6:train.append(np.linalg.norm(d[40:-40],axis=2))
            if r in [8,9]:
                mask=run==r;dyn[wi,mask]=d[idx[mask]-lo]
                old=np.linalg.norm(savgol_filter(kin[lo:hi],21,2,deriv=1,delta=.002,axis=0),axis=1)
                np.testing.assert_allclose(old[idx[mask]-lo],np.array([float(q['speed']) for q in lead])[mask],rtol=1e-10,atol=1e-12)
        train=np.concatenate(train)
        bounds[str(window)]=np.quantile(train,[.01,.99],axis=0).tolist()
        bin_edges[str(window)]=np.r_[0,np.unique(np.quantile(train[:,0],[.2,.4,.6,.8,.9,.95,.99])),np.inf]
        print('DERIVATIVES',window,'targets',n,flush=True)
    fitpack={};oof=np.full((3,6,9,n),np.nan);fitrows=[];trialrows=[];summary={};correlations=[];binned=[];stage_trials=[]
    specs=list(SPECS);runfolds=[(8,9),(9,8)]
    for wi,window in enumerate(WINDOWS):
        g=groups(position,stage,dyn[wi]);speed=np.linalg.norm(dyn[wi,:,0],axis=1);acc=np.linalg.norm(dyn[wi,:,1],axis=1);jerk=np.linalg.norm(dyn[wi,:,2],axis=1)
        Xs={s:np.column_stack([g[c] for c in chars]) if chars else None for s,chars in SPECS.items()}
        summary[str(window)]={}
        for mi,m in enumerate(MODELS):
            y=error[mi];entry={}
            for si,s in enumerate(specs):
                fold=[]
                for train_run,test_run in runfolds:
                    a=primary&(run==train_run);b=primary&(run==test_run);w=trial_weights(keys[a]);prefix=f'w{window}_{m}_{s}_train{train_run}'
                    fitpack[prefix+'_train_indices']=np.flatnonzero(a);fitpack[prefix+'_eval_indices']=np.flatnonzero(b)
                    if s=='constant':
                        intercept=float(np.average(y[a],weights=w));p=np.full(b.sum(),intercept);fitpack[prefix+'_intercept']=np.array(intercept)
                    else:
                        X=Xs[s];scaler=StandardScaler().fit(X[a],sample_weight=w);est=Ridge(alpha=10,solver='svd').fit(scaler.transform(X[a]),y[a],sample_weight=w)
                        p=est.predict(scaler.transform(X[b]));fitpack[prefix+'_coef']=est.coef_;fitpack[prefix+'_intercept']=np.array(est.intercept_);fitpack[prefix+'_mean']=scaler.mean_;fitpack[prefix+'_scale']=scaler.scale_
                    oof[wi,mi,si,b]=p;sc=scores(y[b],p,keys[b]);fold.append({'train_series':train_run,'eval_series':test_run,**sc})
                    fitrows.append({'window_samples':window,'decoder':m,'features':s,'train_series':train_run,'eval_series':test_run,**sc})
                entry[s]={'combined':scores(y[primary],oof[wi,mi,si,primary],keys[primary]),'folds':fold}
                for key in np.unique(keys[primary]):
                    q=primary&(keys==key);res=oof[wi,mi,si,q]-y[q]
                    trialrows.append({'window_samples':window,'decoder':m,'features':s,'series':int(key//100),'trial':int(key%100),'n':int(q.sum()),'mse':float(np.mean(res**2)),'mae':float(np.mean(np.abs(res))),'mean_true_error':float(y[q].mean()),'mean_explained_error':float(oof[wi,mi,si,q].mean())})
            summary[str(window)][m]=entry
            for st in range(1,9):
                q=primary&(stage==st)
                for label,values in [('speed',speed),('velocity_x',dyn[wi,:,0,0]),('velocity_y',dyn[wi,:,0,1]),('velocity_z',dyn[wi,:,0,2]),('acceleration',acc),('jerk',jerk)]:
                    rho=float(spearmanr(values[q],y[q]).statistic) if q.sum()>2 and np.ptp(values[q])>0 and np.ptp(y[q])>0 else None
                    correlations.append({'window_samples':window,'decoder':m,'stage':STAGES[st],'quantity':label,'n':int(q.sum()),'trials':int(len(np.unique(keys[q]))),'spearman_rho':rho})
                for key in np.unique(keys[q]):
                    z=q&(keys==key)
                    stage_trials.append({'window_samples':window,'decoder':m,'series':int(key//100),'trial':int(key%100),'stage':STAGES[st],'weight_code':int(weight[z][0]),'n':int(z.sum()),'pure_window_n':int(pure[z].sum()),'mean_error':float(y[z].mean()),'mean_speed':float(speed[z].mean()),'mean_acceleration':float(acc[z].mean()),'mean_jerk':float(jerk[z].mean())})
            edges=bin_edges[str(window)]
            for st in [0]+list(range(1,9)):
                for bi in range(len(edges)-1):
                    q=primary&(speed>=edges[bi])&(speed<edges[bi+1]);q &= (stage==st) if st else True
                    if not q.any():continue
                    means=[y[q&(keys==k)].mean() for k in np.unique(keys[q])]
                    binned.append({'window_samples':window,'decoder':m,'stage':'all_trial_stages' if st==0 else STAGES[st],'speed_bin':bi,'n':int(q.sum()),'trials':len(means),'median_speed':float(np.median(speed[q])),'trial_mean_error':float(np.mean(means)),'trial_q25':float(np.quantile(means,.25)),'trial_q75':float(np.quantile(means,.75))})
            print('EXPLAINED',window,m,flush=True)
    csvwrite(out/'explanatory_folds.csv',fitrows);csvwrite(out/'explanatory_trials.csv',trialrows);csvwrite(out/'stage_trial_dynamics.csv',stage_trials);csvwrite(out/'within_stage_correlations.csv',correlations);csvwrite(out/'speed_bins.csv',binned)
    wi=WINDOWS.index(41);mi=MODELS.index('cnn_bilstm');ps=specs.index('position_stage');full=specs.index('combined');trialkeys=np.unique(keys[primary]);deltas=[]
    for key in trialkeys:
        q=primary&(keys==key);deltas.append(np.mean((oof[wi,mi,full,q]-error[mi,q])**2)-np.mean((oof[wi,mi,ps,q]-error[mi,q])**2))
    deltas=np.array(deltas);intervals=[block_interval(deltas,trialkeys,b,r) for b,r in [(4,5000),(2,2000),(8,2000)]]
    csvwrite(out/'primary_paired_trials.csv',[{'series':int(k//100),'trial':int(k%100),'combined_minus_position_stage_mse':float(d)} for k,d in zip(trialkeys,deltas)])
    # Region definitions use training-only anchors and fixed residual thresholds.
    edges=np.linspace(0,1,51);anchors=[];region_names=['near_diagonal','observed_anchor_only_high_error','predicted_anchor_only_high_error','both_anchors_high_error','neither_anchor_high_error','intermediate_error'];regionrows=[];regiontrialrows=[];region_codes=np.empty((6,3,n),int)
    trainpos=kin[:starts[6]];speed=np.linalg.norm(dyn[wi,:,0],axis=1);acc=np.linalg.norm(dyn[wi,:,1],axis=1);jerk=np.linalg.norm(dyn[wi,:,2],axis=1)
    for axis in range(3):
        bins=np.argsort(-np.histogram(trainpos[:,axis],edges)[0],kind='stable')[:5];anchors.append(bins.tolist())
        for im,m in enumerate(MODELS):
            y=observed[im,:,axis];p=predicted[im,:,axis];ob=np.clip(np.searchsorted(edges,y,side='right')-1,0,49);pb=np.clip(np.searchsorted(edges,p,side='right')-1,0,49)
            oa=np.isin(ob,bins)&(y>=0)&(y<=1);pa=np.isin(pb,bins)&(p>=0)&(p<=1);e=np.abs(p-y)
            reg=np.full(n,5);reg[e<=.05]=0;high=e>=.1;reg[high&oa&~pa]=1;reg[high&~oa&pa]=2;reg[high&oa&pa]=3;reg[high&~oa&~pa]=4;region_codes[im,axis]=reg
            for code,label in enumerate(region_names):
                q=eligible&(reg==code)
                regionrows.append({'decoder':m,'axis':'XYZ'[axis],'region':label,'n':int(q.sum()),'trial_samples':int((q&primary).sum()),'trials':int(len(np.unique(keys[q&primary]))),'median_speed':float(np.median(speed[q])) if q.any() else None,'low_speed_fraction':float(np.mean(speed[q]<=.02)) if q.any() else None,'median_acceleration':float(np.median(acc[q])) if q.any() else None,'median_jerk':float(np.median(jerk[q])) if q.any() else None,'mean_absolute_axis_error':float(e[q].mean()) if q.any() else None})
                for key in np.unique(keys[q&primary]):
                    z=q&primary&(keys==key);regiontrialrows.append({'decoder':m,'axis':'XYZ'[axis],'region':label,'series':int(key//100),'trial':int(key%100),'n':int(z.sum()),'mean_speed':float(speed[z].mean()),'mean_absolute_axis_error':float(e[z].mean())})
    csvwrite(out/'scatter_regions.csv',regionrows);csvwrite(out/'scatter_region_trials.csv',regiontrialrows)
    np.savez_compressed(out/'analysis_data.npz',target_index=idx,series=run,trial=trial,stage=stage,trial_key=keys,time=time,position=position,observed=observed,predicted=predicted,error=error,derivatives=dyn,eligible=eligible,primary=primary,oof=oof,pure_window=pure,weight=weight,region_codes=region_codes)
    np.savez_compressed(out/'fitted_explainers.npz',**fitpack)
    leadsummary=summary['41']['cnn_bilstm'];base=leadsummary['position_stage']['combined']['mse'];allm=leadsummary['combined']['combined']['mse']
    info={'experiment_id':'EXP-036','status':'complete','created_at':datetime.now(timezone.utc).isoformat(),'participant':'P1','series':[8,9],'models':MODELS,'lead_model':'cnn_bilstm','smoothing_windows_samples':WINDOWS,'primary_window_samples':41,'sampling_hz':500,'ridge_alpha':10,'feature_specs':SPECS,'n_saved_targets_per_model':n,'n_eligible_targets':int(eligible.sum()),'n_primary_targets':int(primary.sum()),'n_primary_trials':68,'decoder_training_or_inference':False,'explanatory_fit_count':len(fitrows),'trial_balanced_metrics':True,'source_hashes':sourcehash,'snapshot_hashes':{name:sha(out/name) for name in snapshots},'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'train_colour_bounds':bounds,'anchors_by_axis':anchors,'scores':summary,'primary_contrast':{'description':'combined minus position_stage explanatory MSE; negative favours dynamics','mean_paired_trial_delta':float(deltas.mean()),'trials_improved':int((deltas<0).sum()),'mse_reduction_percent':float(100*(base-allm)/base),'intervals':intervals},'limits':['Retrospective use of observed kinematics and familiar P1 test series.','Explanatory fits train on one series and evaluate on the other; they do not retrain or correct wrist decoders.','Conditional pointwise trial-block intervals omit fitting/population uncertainty.','Symmetric derivatives and unknown upstream scaling/filtering do not establish real-time or physical/neural causality.','Jerk and exact geometric region thresholds are exploratory diagnostics.']}
    assert all(sha(ROOT/p if not p.startswith('../') else ROOT.parent/p[3:])==h for p,h in sourcehash.items());info['source_files_unchanged']=True
    dump(out/'results.json',info)
    make_plots(out,info,observed,predicted,error,dyn,primary,stage,eligible,region_codes,binned,summary)
    report(out,info,regionrows)
    print('COMPLETE',out,flush=True)

def make_plots(out,info,observed,predicted,error,dyn,primary,stage,eligible,regions,binned,summary):
    mi=MODELS.index('cnn_bilstm');wi=WINDOWS.index(41);names=['Speed','Acceleration norm','Jerk norm'];q=primary;bounds=np.array(info['train_colour_bounds']['41'])
    fig,ax=plt.subplots(3,3,figsize=(16,13),layout='constrained')
    for a in range(3):
        for d in range(3):
            c=np.linalg.norm(dyn[wi,:,d],axis=1);lo=max(bounds[0,d],1e-9);hi=max(bounds[1,d],lo*10)
            h=ax[a,d].scatter(observed[mi,q,a],predicted[mi,q,a],c=np.maximum(c[q],lo),norm=LogNorm(lo,hi,clip=True),cmap='viridis',s=7,alpha=.55,rasterized=True)
            v=np.r_[observed[mi,q,a],predicted[mi,q,a]];low,high=v.min(),v.max();pad=.04*(high-low);ax[a,d].plot([low-pad,high+pad],[low-pad,high+pad],'k--',lw=.7);ax[a,d].set(xlim=(low-pad,high+pad),ylim=(low-pad,high+pad),xlabel='Observed '+ 'XYZ'[a]+' (reduced)',ylabel='Predicted '+ 'XYZ'[a]+' (reduced)');ax[a,d].set_aspect('equal',adjustable='box');ax[a,d].set_title(names[d]);fig.colorbar(h,ax=ax[a,d],shrink=.7,label=f'reduced units/s^{d+1}')
    fig.suptitle('Saved CNN–BiLSTM scatter coloured by observed dynamics\nP1 trial targets, 82-ms cubic smoothing; colour limits from series 1–6',fontsize=14)
    finishfig(fig,out/'scatter_by_dynamics.png')
    fig,ax=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for d in range(3):
        c=np.linalg.norm(dyn[wi,:,d],axis=1);ax[d].scatter(np.maximum(c[q],1e-9),error[mi,q],c=stage[q],cmap='tab10',s=6,alpha=.35,rasterized=True);ax[d].set(xscale='log',xlabel=names[d]+f' (reduced units/s^{d+1})',ylabel='Original XYZ error norm (reduced)',title=names[d]+' versus decoder error')
    fig.suptitle('Observed dynamics and saved decoder errors; dots are correlated trial samples')
    finishfig(fig,out/'error_vs_dynamics.png')
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    for st,ax in zip(STAGES[1:],axes.flat):
        rr=[r for r in binned if r['window_samples']==41 and r['decoder']=='cnn_bilstm' and r['stage']==st and r['trials']>=5]
        if rr:
            xx=[max(r['median_speed'],1e-8) for r in rr];yy=[r['trial_mean_error'] for r in rr];ax.plot(xx,yy,'o-',lw=1.5);ax.vlines(xx,[r['trial_q25'] for r in rr],[r['trial_q75'] for r in rr],lw=2,alpha=.5)
        else:ax.text(.5,.5,'Insufficient bin coverage',ha='center',transform=ax.transAxes)
        ax.set(xscale='log',title=st.replace('_',' '),xlabel='Speed (reduced units/s)',ylabel='Trial-mean error')
    fig.suptitle('Within-stage speed curves for CNN–BiLSTM\nTraining-defined bins, equal contributing-trial means, IQR bars; at least 5 trials/bin')
    finishfig(fig,out/'within_stage_speed.png')
    specs=list(SPECS);mat=np.array([[summary['41'][m][s]['combined']['r2_error_target'] for s in specs] for m in MODELS])
    fig,ax=plt.subplots(figsize=(14,5),layout='constrained');h=ax.imshow(mat,cmap='RdYlGn',vmin=-.5,vmax=.8,aspect='auto');ax.set_xticks(np.arange(9),['Constant','Position','Stage','Dynamics','P + S','P + D','S + D','P + S + D','P + S + D + J'],rotation=25,ha='right');ax.set_yticks(np.arange(6),MODELS)
    for a in range(6):
        for b in range(9):ax.text(b,a,f'{mat[a,b]:.2f}',ha='center',va='center',fontsize=10)
    fig.colorbar(h,ax=ax,label='Out-of-series error-target R²');ax.set_title('Explaining decoder error across series (trial-balanced, 82-ms smoothing)\nThese R² values describe prediction error, not wrist reconstruction')
    finishfig(fig,out/'explanatory_comparison.png')
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    for w,offset in [(21,-.2),(41,0),(81,.2)]:
        values=[100*(summary[str(w)][m]['position_stage']['combined']['mse']-summary[str(w)][m]['combined']['combined']['mse'])/summary[str(w)][m]['position_stage']['combined']['mse'] for m in MODELS]
        ax.bar(np.arange(6)+offset,values,.19,label=f'{w*2}-ms smoothing')
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(np.arange(6),MODELS,rotation=15);ax.set_ylabel('Explanatory MSE reduction from adding dynamics (%)');ax.set_title('Incremental dynamics information beyond position and stage\nPositive favours P + S + D over P + S; cross-series, trial-balanced');ax.legend()
    finishfig(fig,out/'incremental_dynamics.png')
    colors=['#228B68','#D95F02','#756BB1','#E7298A','#888888','#D3D3D3'];labels=['Near diagonal','Observed anchor only, high error','Predicted anchor only, high error','Both anchors, high error','Neither anchor, high error','Intermediate error']
    fig,ax=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for a in range(3):
        for code,label in enumerate(labels):
            z=eligible&(regions[mi,a]==code);ax[a].scatter(observed[mi,z,a],predicted[mi,z,a],s=6,alpha=.5,color=colors[code],label=label)
        v=np.r_[observed[mi,eligible,a],predicted[mi,eligible,a]];lo,hi=v.min(),v.max();ax[a].plot([lo,hi],[lo,hi],'k--',lw=.7);ax[a].set(xlabel='Observed '+'XYZ'[a],ylabel='Predicted '+'XYZ'[a]);ax[a].set_aspect('equal',adjustable='box')
    handles,labels=ax[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.13),ncol=3,fontsize=9);fig.suptitle('CNN–BiLSTM scatter regions: fixed geometry, training-only position anchors\nAll eligible targets including intertrial samples; region/error association is descriptive')
    finishfig(fig,out/'scatter_regions.png')

def report(out,d,regions):
    p=d['primary_contrast'];ci=p['intervals'][0];s=d['scores']['41']['cnn_bilstm'];lines=['# EXP-036: movement dynamics and saved-decoder error','',f'P1 series 8–9, six fixed decoders, {d["n_primary_targets"]} trial targets across 68 trials. No decoder training or inference. Error explainers fit on one series and evaluate on the other.','', '## Main comparison','',f'Adding dynamics to position and stage changes CNN–BiLSTM error-explanation MSE by {p["mse_reduction_percent"]:.2f}% reduction. Combined-minus-position/stage paired trial loss: {p["mean_paired_trial_delta"]:.6f}; {p["trials_improved"]}/68 trials improve.',f'Exploratory 95% block-4 interval: [{ci["lower_95"]:.6f}, {ci["upper_95"]:.6f}]. This interval is conditional on the fitted explainers and familiar P1 series.','', '| Features | Explanatory MSE | Error-target R² |','|---|---:|---:|']
    for name,e in s.items():lines.append(f'| {name} | {e["combined"]["mse"]:.6f} | {e["combined"]["r2_error_target"]:.3f} |')
    lines+=['','These are scores for explaining the decoder’s error magnitude, not improved wrist-decoding scores. Kinematic and stage features use observed information retrospectively.','', '![Added dynamics and smoothing sensitivity](incremental_dynamics.png)','','![All six explanatory comparisons](explanatory_comparison.png)','','## Movement dynamics and scatter structure','','![Scatter coloured by dynamics](scatter_by_dynamics.png)','','![Dynamics versus error](error_vs_dynamics.png)','','![Within-stage speed](within_stage_speed.png)','','![Geometric scatter regions](scatter_regions.png)','','| Axis / region | Samples | Median speed | Low-speed fraction |','|---|---:|---:|---:|']
    for r in regions:
        if r['decoder']=='cnn_bilstm' and r['region'] in ['near_diagonal','observed_anchor_only_high_error','predicted_anchor_only_high_error'] and r['n']:
            lines.append(f'| {r["axis"]} / {r["region"]} | {r["n"]} | {r["median_speed"]:.4f} | {r["low_speed_fraction"]:.1%} |')
    lines+=['','## Interpretation and limits','','The curves, explanatory contrasts and smoothing sensitivity answer related but different questions. Inspect each series and stage rather than assuming a single speed threshold explains all scatter streaks. Geometry-based regions are descriptive because their definitions already involve prediction error. Jerk adds a smoothing-sensitive diagnostic.','', 'No phase-specific kinematics decoder, EEG stage classifier, new segmentation or online error predictor was trained. Existing test series have informed these hypotheses; all findings are exploratory. Source predictions and prior reports remain unchanged.','', '## Evidence','','- `results.json`: protocol/source hashes, all feature-set scores, fold scores and primary uncertainty.','- `explanatory_trials.csv`, `primary_paired_trials.csv`: trial-level paired losses.','- `stage_trial_dynamics.csv`, `within_stage_correlations.csv`, `speed_bins.csv`: stage coverage and descriptive associations.','- `scatter_regions.csv`, `scatter_region_trials.csv`: geometric-region composition.','- `analysis_data.npz`, `fitted_explainers.npz`: local replay arrays and coefficients, ignored by Git.','- Exact source/protocol snapshots retained; independent verification is recorded separately.','','[Declared protocol](../../documentation/methods/ERROR_DYNAMICS.md) · [Registry](../../documentation/EXPERIMENT_REGISTRY.md)','']
    (out/'RESULTS.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
