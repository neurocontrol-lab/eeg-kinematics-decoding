"""LEDOn-aligned trial averages and exploratory simultaneous error bands.

Read saved P1 trace data only. Resample consecutive trial blocks, never windows.
High error means above each trial's own mean preparation-period error.
"""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import argparse, csv, hashlib, json
from pathlib import Path
import sys
_repo = next(p for p in Path(__file__).resolve().parents if (p / "src/eeg_kinematics/paths.py").is_file())
sys.path.insert(0, str(_repo / "src"))
from eeg_kinematics.paths import REPO_ROOT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=REPO_ROOT
NAMES=['training_mean','ridge','cnn','cnn_bilstm','cnn_elu','eegnet_regression']
STAGE_LABELS=['Preparation','Reach','Preload','Loading','Lift + hold','Cue to replacement','Release','Return']

def mean(a,axis=0):
    n=np.isfinite(a).sum(axis=axis)
    return np.divide(np.nansum(a,axis=axis),n,out=np.full(np.shape(n),np.nan,dtype=float),where=n>0)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def spans(mask,grid,step):
    inds=np.flatnonzero(mask)
    if not len(inds):return []
    groups=np.split(inds,np.flatnonzero(np.diff(inds)>1)+1)
    return [(float(max(0,grid[g[0]]-step/2)),float(grid[g[-1]]+step/2),g) for g in groups]

def bootstrap(runs,block,b,seed):
    rng=np.random.default_rng(seed);res={}
    for r,data in runs.items():
        n=data['error'].shape[0];t=data['error'].shape[-1]
        bs=rng.integers(n,size=(b,int(np.ceil(n/block))))
        indices=((bs[:,:,None]+np.arange(block))%n).reshape(b,-1)[:,:n]
        errors=np.empty((b,len(NAMES),t));diffs=np.empty_like(errors)
        for begin in range(0,b,100):
            ix=indices[begin:begin+100]
            errors[begin:begin+len(ix)]=mean(data['error'][ix],axis=1)
            diffs[begin:begin+len(ix)]=mean(data['difference'][ix],axis=1)
        res[r]={'error':errors,'difference':diffs,'se':np.nanstd(diffs,axis=0,ddof=1)}
    maximum=np.zeros(b)
    for r,data in runs.items():
        se=res[r]['se'];ok=(data['coverage']>=data['min_trials'])&(se>1e-10)
        centered=np.abs((res[r]['difference']-data['mean_difference'])/np.where(se>1e-10,se,np.nan))
        centered[:,~ok]=np.nan
        maximum=np.maximum(maximum,np.nanmax(centered.reshape(b,-1),axis=1))
        res[r]['eligible']=ok
    critical=float(np.quantile(maximum,.95))
    for r,data in runs.items():
        z=res[r];z['low']=data['mean_difference']-critical*z['se'];z['high']=data['mean_difference']+critical*z['se']
        z['significant']=(z['low']>0)&z['eligible']
        z['error_low'],z['error_high']=np.nanquantile(z['error'],[.025,.975],axis=0)
    return critical,res

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('analysis_dir',type=Path)
    p.add_argument('--resamples',type=int,default=5000)
    args=p.parse_args();src=args.analysis_dir.resolve()
    assert src.parent==(ROOT/'outputs').resolve()
    out=src/'trial_averages';out.mkdir(exist_ok=True)
    sources=[src/'aligned_predictions.csv',src/'trial_events.json',src/'results.json']
    hashes={q.name:digest(q) for q in sources}
    events=json.loads(sources[1].read_text());rows=list(csv.DictReader(sources[0].open()))
    assert {r['model'] for r in rows}==set(NAMES), 'Trace rows must contain exactly the six reference models'
    step=.128
    end=max(q['events'][-1]-q['events'][0] for ts in events.values() for q in ts)
    grid=np.arange(int(np.ceil(end/step))+1)*step
    runs={}
    for run in [8,9]:
        ev=events[str(run)];n=len(ev);assert n==34
        shape=(n,len(NAMES),len(grid))
        obs=np.full(shape+(3,),np.nan);pred=np.full_like(obs,np.nan);err=np.full(shape,np.nan)
        baseline=np.full((n,len(NAMES)),np.nan);base_counts=np.zeros_like(baseline,dtype=int)
        for mi,name in enumerate(NAMES):
            for ti,tr in enumerate(ev):
                group=[x for x in rows if x['model']==name and int(x['run'])==run and int(x['trial'])==tr['trial']]
                assert group
                t=np.array([float(x['run_time_seconds'])-tr['events'][0] for x in group])
                y=np.array([[float(x['y_'+a]) for a in ['x','y','z']] for x in group])
                yp=np.array([[float(x['pred_'+a]) for a in ['x','y','z']] for x in group])
                e=np.linalg.norm(yp-y,axis=1)
                prep=np.array([x['stage']=='preparation' for x in group]);assert prep.any()
                baseline[ti,mi]=e[prep].mean();base_counts[ti,mi]=prep.sum()
                # At most one original target per nearest-centred 128-ms bin.
                bins=np.floor(t/step+.5).astype(int);assert len(np.unique(bins))==len(bins)
                assert np.all(t>=-1e-9) and np.all(t<tr['events'][-1]-tr['events'][0]+1e-9)
                obs[ti,mi,bins]=y;pred[ti,mi,bins]=yp;err[ti,mi,bins]=e
        difference=err-baseline[:,:,None]
        runs[run]={'observed':obs,'predicted':pred,'error':err,'baseline':baseline,'baseline_counts':base_counts,
                   'difference':difference,'mean_difference':mean(difference),'coverage':np.isfinite(err).sum(axis=0),
                   'min_trials':int(np.ceil(.8*n)),'events':np.median([np.array(tr['events'])-tr['events'][0] for tr in ev],axis=0)}
    print('ALIGNED',len(rows),'trace rows; 34 trials in each run',flush=True)
    critical,primary=bootstrap(runs,4,args.resamples,42)
    alternatives={}
    for length in [2,8]:
        q,z=bootstrap(runs,length,2000,42+length);alternatives[length]=(q,z)
    print('SIMULTANEOUS_CRITICAL',critical,flush=True)
    report={'status':'complete','participant':'P1','runs':[8,9],'models':NAMES,'trials_per_run':34,'bin_seconds':step,
       'alignment':'LEDOn = 0 seconds; nearest 128-ms bin; no interpolation/extrapolation and at most one source target per trial/bin',
       'minimum_coverage':28,'baseline':'Each trial and model own mean Euclidean prediction error during event-defined preparation',
       'error_units':'Euclidean distance in reduced target coordinates, not physical distance',
       'bootstrap':{'resamples':args.resamples,'seed':42,'primary_block_length_trials':4,'sensitivity_block_lengths_trials':[2,8],
                    'sensitivity_resamples':2000,'interval':'Exploratory simultaneous 95% centered standardized max-statistic bootstrap band',
                    'multiplicity_family':'All eligible time bins, all six model outputs, and both runs together',
                    'critical_value':critical,'sensitivity_critical_values':{str(k):v[0] for k,v in alternatives.items()}},
       'limits':['Exploratory inference after observing these held-out runs; not a preregistered confirmatory test.',
                 '34 consecutive trials per run and one participant limit precision and generalization.',
                 'Circular moving-block trial resampling approximates local serial dependence; stationarity is an assumption.',
                 'Highlighted intervals mean higher error than preparation, not error greater than zero or need for a different decoder.',
                 'Trials vary in weight and timing. LEDOn alignment can blur later stages; background boundaries are medians.',
                 'Short stages and source sampling cadence limit temporal precision. Excluded low-coverage bins are not tested.',
                 'The norm is computed per original prediction before averaging; it is not the norm of the averaged residual.'],
       'source_sha256':hashes,'analysis_sha256':digest(Path(__file__)),'results':{}}
    curve_rows=[]
    for run,data in runs.items():
        report['results'][str(run)]={}
        for mi,name in enumerate(NAMES):
            z=primary[run];sig=z['significant'][mi]
            robust=sig.copy()
            for _,alt in alternatives.values():robust &= alt[run]['significant'][mi]
            regions=[]
            for a,b,ix in spans(sig,grid,step):
                regions.append({'start_seconds_from_LEDOn':a,'end_seconds_from_LEDOn':b,'bins':len(ix),
                  'min_trials':int(data['coverage'][mi,ix].min()),'mean_excess_error':float(np.mean(data['mean_difference'][mi,ix])),
                  'minimum_simultaneous_lower_bound':float(np.min(z['low'][mi,ix])),
                  'all_bins_also_positive_with_blocks_2_and_8':bool(robust[ix].all())})
            report['results'][str(run)][name]={'preparation_mean_error':float(data['baseline'][:,mi].mean()),
              'preparation_samples_per_trial':data['baseline_counts'][:,mi].tolist(),'regions':regions,
              'primary_significant_bins':int(sig.sum()),'robust_bins_all_block_lengths':int(robust.sum())}
            y=mean(data['observed'][:,mi]);yp=mean(data['predicted'][:,mi]);ee=mean(data['error'][:,mi])
            for j,t in enumerate(grid):
                if data['coverage'][mi,j]:curve_rows.append([run,name,t,int(data['coverage'][mi,j]),*y[j],*yp[j],ee[j],data['mean_difference'][mi,j],z['low'][mi,j],z['high'][mi,j],bool(z['eligible'][mi,j]),bool(sig[j]),bool(robust[j])])
            fig,ax=plt.subplots(6,1,figsize=(13,13),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1,1,1,1.1,1.1,.4]})
            colors=plt.cm.Set3(np.linspace(0,1,9));med=data['events']
            for a in range(3):
                for si in range(1,9):ax[a].axvspan(med[si-1],med[si],color=colors[si],alpha=.18)
                oq=np.nanquantile(data['observed'][:,mi],[.25,.75],axis=0);pq=np.nanquantile(data['predicted'][:,mi],[.25,.75],axis=0)
                ax[a].fill_between(grid,oq[0,:,a],oq[1,:,a],color='black',alpha=.10,label='Observed trial IQR')
                ax[a].fill_between(grid,pq[0,:,a],pq[1,:,a],color='tab:blue',alpha=.14,label='Predicted trial IQR')
                ax[a].plot(grid,y[:,a],'k.-',label='Mean observed');ax[a].plot(grid,yp[:,a],'.-',color='tab:blue',label='Mean predicted')
                ax[a].set_ylabel('XYZ'[a]+' (reduced)')
            ax[0].legend(ncol=4,fontsize=9,loc='upper right')
            ax[3].plot(grid,ee,color='tab:red',label='Mean trial error norm');ax[3].fill_between(grid,z['error_low'][mi],z['error_high'][mi],color='tab:red',alpha=.15,label='Pointwise 95% trial-block interval')
            ax[3].axhline(data['baseline'][:,mi].mean(),color='gray',ls='--',label='Mean preparation error')
            ax[3].set_ylabel('Mean error norm');ax[3].legend(ncol=3,fontsize=8)
            dd=data['mean_difference'][mi];eligible=z['eligible'][mi]
            ax[4].plot(grid,np.where(eligible,dd,np.nan),color='black',label='Mean paired excess over preparation')
            ax[4].fill_between(grid,np.where(eligible,z['low'][mi],np.nan),np.where(eligible,z['high'][mi],np.nan),color='gray',alpha=.25,label='Simultaneous 95% band')
            ax[4].axhline(0,color='black',ls='--',lw=.7);ax[4].set_ylabel('Error − preparation')
            for a,b,_ in spans(sig,grid,step):
                ax[3].axvspan(a,b,color='orange',alpha=.25);ax[4].axvspan(a,b,color='orange',alpha=.25)
            ax[4].legend(fontsize=8,ncol=2)
            ax[5].step(grid,data['coverage'][mi],where='mid',color='tab:green');ax[5].axhline(28,color='gray',ls='--',lw=.7)
            ax[5].set(ylim=(0,36),yticks=[0,28,34],ylabel='Trials',xlabel='Seconds from LEDOn (128-ms bins)')
            xmax=grid[data['coverage'][mi]>0].max()+step/2
            ax[-1].set_xlim(0,xmax)
            for a in ax:a.grid(alpha=.15)
            fig.suptitle(f'P1 run {run}: {name}, average over 34 trials\nOrange: exploratory elevated-error intervals vs preparation; corrected across all models/runs/time bins')
            fig.legend(handles=[Patch(facecolor=colors[si],alpha=.4,label=STAGE_LABELS[si-1]) for si in range(1,9)],loc='lower center',bbox_to_anchor=(.5,-.085),ncol=4,fontsize=9,title='Median stage intervals (individual trial timing varies)')
            fig.savefig(out/f'run_{run}_{name}_average.png',dpi=140,bbox_inches='tight');plt.close(fig)
            print('AVERAGE',run,name,'high-error bins',sig.sum(),'robust',robust.sum(),flush=True)
    with (out/'average_curves.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['run','model','seconds_from_LEDOn','trials','observed_x','observed_y','observed_z','predicted_x','predicted_y','predicted_z','mean_error_norm','mean_paired_excess','simultaneous_low','simultaneous_high','eligible','elevated','robust_blocks_2_4_8']);w.writerows(curve_rows)
    np.savez_compressed(out/'trial_arrays.npz',grid=grid,**{f'run{r}_{k}':v for r,d in runs.items() for k,v in d.items() if isinstance(v,np.ndarray)})
    assert hashes=={q.name:digest(q) for q in sources}
    report['source_files_unchanged']=True
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    lines=['# LEDOn-aligned trial averages','',
      'Each run contributes 34 held-out P1 trials. Existing run-wide and individual-trial plots are retained.',
      'Coordinates are trial means; shading shows across-trial interquartile ranges. Error is the mean of individual prediction error norms, so opposite signed errors cannot cancel.',
      'Original predictions are assigned to nearest 128-ms LEDOn-relative bins. No new inference, interpolation or extrapolation occurs. Median event boundaries are guides, not identical stage timings for all trials.','',
      '## Exploratory elevated-error intervals','',
      'For each trial/model, subtract its own mean preparation-period error. Orange regions have a simultaneous 95% lower confidence bound above zero, using circular moving-block bootstrap over consecutive trials (block size 4). The maximum standardized bootstrap deviation corrects jointly across eligible bins, all six model outputs and both runs. Only bins supported by at least 28/34 trials are tested.',
      'This tests higher error than preparation, not greater-than-zero error or the need for separate stage decoders. It is an exploratory analysis of one participant. Block sizes 2 and 8 provide sensitivity checks.','',
      '| Run | Model | Seconds from LEDOn | Minimum trials | Mean excess error | Positive throughout at blocks 2, 4 and 8? |','|---|---|---|---:|---:|---|']
    for run in [8,9]:
        for name in NAMES:
            for q in report['results'][str(run)][name]['regions']:
                lines.append(f'| {run} | {name} | [{q["start_seconds_from_LEDOn"]:.3f}, {q["end_seconds_from_LEDOn"]:.3f}) | {q["min_trials"]} | {q["mean_excess_error"]:.4f} | {"Yes" if q["all_bins_also_positive_with_blocks_2_and_8"] else "Not every bin"} |')
    lines+=['','## CNN–BiLSTM averages','', '![Run 8 average](run_8_cnn_bilstm_average.png)','','![Run 9 average](run_9_cnn_bilstm_average.png)','',
       '## Files and limitations','',
       'summary.json records the reference, bootstrap settings, multiplicity family, coverage and interval endpoints. average_curves.csv contains mean curves and simultaneous bands. Trial arrays remain local in trial_arrays.npz. Plots are also available for the other five saved outputs.',
       'Pointwise intervals on the mean-error panel describe uncertainty in the mean. Only the simultaneous paired-excess band determines orange highlighting.',
       'Changing trial duration/weight can affect the average and blur late phases. Preparation estimates can use few source points. Consecutive-trial resampling assumes approximate local stationarity; block-size sensitivity does not establish that assumption. These plots are not stage-trained decoder comparisons, population inference, or proof of different neural routines.','']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
