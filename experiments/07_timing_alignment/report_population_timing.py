"""Publish participant-first EXP-042 timing tables and scientific plots."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/eeg_kinematics/paths.py').is_file());sys.path.insert(0,str(ROOT/'src'))
from eeg_kinematics.population_timing import summarize_population
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

LABELS={'ridge':'Ridge','cnn_elu':'CNN + ELU'}
COLORS={'ridge':'#3E6F8C','cnn_elu':'#B15F26'}

def csvwrite(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output_dir',type=Path);a=ap.parse_args();out=a.output_dir.resolve()
    assert out.parent==(ROOT/'outputs').resolve()
    d=json.loads((out/'results.json').read_text());s=summarize_population(d)
    (out/'population_summary.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    rows=s['participant_metrics'];people=d['participants'];gaps=d['gap_ms'];models=['ridge','cnn_elu']
    flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows];csvwrite(out/'participant_metrics.csv',flat)
    valrows=[];seedrows=[]
    for person in people:
        e=d['participant_results'][person]
        for gap,g in e['gaps'].items():
            for c in g['ridge']['candidates']:valrows.append({'participant':person,'model':'ridge','gap_ms':int(gap)*2,'seed':'','alpha':c['alpha'],'validation_mse':c['validation']['mse_mean'],'selected_alpha_at_gap':c['alpha']==g['ridge']['selected_alpha']})
            for seed,c in g['cnn_elu'].items():valrows.append({'participant':person,'model':'cnn_elu','gap_ms':int(gap)*2,'seed':seed,'alpha':'','validation_mse':c['validation']['mse_mean'],'selected_alpha_at_gap':''})
        for r in [r for r in rows if r['participant']==person]:
            for i,delta in enumerate(r['seed_mse_differences']):seedrows.append({'participant':person,'model':r['model'],'seed':d['cnn_seeds'][i] if r['model']=='cnn_elu' else '','selected_gap_ms':r['selected_gap_ms'],'selected_minus_baseline_mse':delta,'percent_mse_reduction':r['seed_percent_reductions'][i]})
    csvwrite(out/'validation_candidates.csv',valrows);csvwrite(out/'seed_comparisons.csv',seedrows)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    def export(fig,name):fig.savefig(out/name,dpi=180,bbox_inches='tight');plt.close(fig)
    val={}
    for model in models:
        val[model]=np.array([[d['participant_results'][p]['gaps'][str(g//2)][model]['validation']['mse_mean'] if model=='ridge' else np.mean([z['validation']['mse_mean'] for z in d['participant_results'][p]['gaps'][str(g//2)][model].values()]) for g in gaps] for p in people])
    fig,axes=plt.subplots(1,2,figsize=(13,8),layout='constrained')
    for ax,model in zip(axes,models):
        pct=100*(val[model]/val[model][:,[0]]-1)
        # Colour saturation is disclosed; annotations retain actual values.
        im=ax.imshow(pct,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-100,vcenter=0,vmax=100))
        for i,p in enumerate(people):
            best=gaps.index(d['selection'][p][model]['gap_ms'])
            for j in range(6):ax.text(j,i,f'{pct[i,j]:+.0f}%'+('*' if j==best else ''),ha='center',va='center',fontsize=9,color='white' if abs(pct[i,j])>60 else 'black')
        ax.set(xticks=range(6),xticklabels=gaps,yticks=range(12),yticklabels=people,xlabel='EEG end gap (ms)',title=LABELS[model])
        fig.colorbar(im,ax=ax,label='Validation MSE change from 2 ms (%)',shrink=.7,extend='both')
    fig.suptitle('Population timing: validation only\nCNN averages three seeds. * selected gap. Colours saturate at ±100%; labels show actual change.')
    export(fig,'validation_heatmaps.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,model in zip(axes,models):
        norm=val[model]/val[model][:,[0]]
        for i,row in enumerate(norm):ax.plot(gaps,row,alpha=.32,lw=1)
        ax.plot(gaps,np.median(norm,axis=0),color='black',marker='o',lw=2.5,label='Participant median')
        ax.axhline(1,color='grey',ls='--');ax.set(xscale='log',yscale='log',xticks=gaps,xticklabels=gaps,xlabel='EEG end gap (ms, log scale)',ylabel='Validation MSE / own 2-ms MSE (log scale)',title=LABELS[model]);ax.legend()
    fig.suptitle('Each thin line is one participant; selections use individual validation scores')
    export(fig,'validation_curves.png')
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    for ax,model in zip(axes,models):
        rr=[r for r in rows if r['model']==model]
        for i,r in enumerate(rr):
            color=plt.cm.tab20(i/12);ax.plot([0,1],[r['baseline_mse'],r['selected_mse']],marker='o',color=color,alpha=.8)
            ax.annotate(r['participant'],(1,r['selected_mse']),xytext=(7,0),textcoords='offset points',fontsize=9,color=color)
        ps=s['population'][model];ax.plot([0,1],[ps['mean_baseline_mse'],ps['mean_selected_mse']],color='black',lw=3,marker='s',label='Equal-participant mean')
        ax.set(xticks=[0,1],xticklabels=['2-ms reference','Validation-selected gap'],ylabel='Test XYZ MSE (training-standardized)',title=LABELS[model]);ax.set_xlim(-.15,1.3);ax.legend(fontsize=9)
    fig.suptitle('Paired test scores for the same targets\nCNN averages three fitted-seed scores within participant, without ensembling predictions')
    export(fig,'paired_test_scores.png')
    fig,axes=plt.subplots(1,2,figsize=(12,7),layout='constrained')
    for ax,model in zip(axes,models):
        rr=[r for r in rows if r['model']==model]
        for i,r in enumerate(rr):
            ax.scatter(r['seed_percent_reductions'],np.repeat(i,len(r['seed_percent_reductions'])),s=20,color=COLORS[model],alpha=.4)
            ax.scatter(r['percent_mse_reduction'],i,s=54,marker='D',color=COLORS[model])
        ps=s['population'][model];lo,hi=ps['percent_reduction_95_interval'];mean=ps['mean_participant_percent_reduction']
        ax.errorbar(mean,13,xerr=[[mean-lo],[hi-mean]],color='black',marker='s',capsize=4,lw=2)
        ax.axvline(0,color='grey',ls='--');ax.set(yticks=list(range(12))+[13],yticklabels=people+['Mean + 95% interval'],xlabel='Test MSE reduction from 2-ms reference (%)\nPositive favours selected gap',title=LABELS[model]);ax.invert_yaxis()
    fig.suptitle('Each participant contributes once\nDiamonds: participant change from seed-mean MSE. Small dots: individual seeds. Intervals resample participants.')
    export(fig,'participant_improvements.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,model in zip(axes,models):
        counts=[s['population'][model]['selected_gap_counts'][str(g)] for g in gaps]
        bars=ax.bar(range(6),counts,color=COLORS[model]);ax.bar_label(bars,padding=3);ax.set(xticks=range(6),xticklabels=gaps,yticks=range(0,13,2),ylim=(0,13),xlabel='Validation-selected end gap (ms)',ylabel='Participants (total 12)',title=LABELS[model])
    fig.suptitle('One selected gap per participant and model; CNN selection averages validation MSE over three seeds')
    export(fig,'selected_gap_counts.png')
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ri,model in enumerate(models):
        rr=[r for r in rows if r['model']==model]
        for a,axis in enumerate('XYZ'):
            ax=axes[ri,a]
            for i,r in enumerate(rr):ax.plot([0,1],[r['baseline_axis_r2'][a],r['selected_axis_r2'][a]],alpha=.5,color=plt.cm.tab20(i/12),marker='.',ms=4)
            ax.axhline(0,color='grey',ls=':');ax.set(xticks=[0,1],xticklabels=['2 ms','Selected'],ylabel='Within-participant test R²',title=LABELS[model]+' / '+axis)
    fig.suptitle('Per-axis test performance, all 12 participants\nLines are participants; CNN scores average the three fits within participant. R² can be negative.')
    export(fig,'axis_test_r2.png')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,model in zip(axes,models):
        rr=[r for r in rows if r['model']==model]
        ax.barh(people,[r['percent_mse_reduction'] for r in rr],color=[COLORS[model] if r['percent_mse_reduction']>=0 else '#9D4B4B' for r in rr]);ax.axvline(0,color='black',lw=1);ax.invert_yaxis();ax.set(xlabel='Test MSE reduction (%)\nPositive favours validation-selected gap',title=LABELS[model])
    fig.suptitle('Population T1: fixed 256-ms windows, six end gaps, participant-specific decoders')
    export(fig,'meeting_summary.png')
    lines=['# EXP-042 / T1: population timing alignment','', 'Twelve participant-specific decoders per model, six end gaps, fixed 256-ms EEG inputs. CNN + ELU uses three seeds; ridge uses five validation-selected alpha candidates per gap. All gap choices froze across all participants before new test scoring.','', '## Population result','', 'Scores average CNN fits within each participant, then give each participant equal weight. These are repeated-fit scores, not an ensemble. Negative paired MSE differences favour the selected gap.','', '| Model | Mean reference MSE | Mean selected MSE | Mean paired difference | Exploratory 95% interval | Improved / unchanged / worse |','|---|---:|---:|---:|---|---|']
    for model in models:
        q=s['population'][model];lo,hi=q['paired_mse_difference_95_interval'];lines.append(f"| {LABELS[model]} | {q['mean_baseline_mse']:.6f} | {q['mean_selected_mse']:.6f} | {q['mean_selected_minus_baseline']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {q['participants_improved']} / {q['participants_unchanged']} / {q['participants_worse']} |")
    lines+=['', 'Intervals resample the 12 participants, after averaging CNN seeds. They condition on the fitted models and fixed split. They are exploratory and do not establish representativeness of a broader human population. A selected 2-ms gap yields exactly zero change.','', '![Population summary](meeting_summary.png)','', '## Participant results']
    for model in models:
        lines+=['', '### '+LABELS[model], '', '| Participant | Selected gap (ms) | Reference MSE | Selected MSE | MSE reduction (%) |','|---|---:|---:|---:|---:|']
        for r in [r for r in rows if r['model']==model]:lines.append(f"| {r['participant']} | {r['selected_gap_ms']} | {r['baseline_mse']:.6f} | {r['selected_mse']:.6f} | {r['percent_mse_reduction']:+.2f} |")
    lines+=['','## Population figures','','![Validation heatmaps](validation_heatmaps.png)','','![Validation curves](validation_curves.png)','','![Paired test scores](paired_test_scores.png)','','![Participant improvement and seed variation](participant_improvements.png)','','![Selected gap distribution](selected_gap_counts.png)','','![Axis-wise test R2](axis_test_r2.png)','','Validation figures show all six gaps. Test figures show only validation-selected gaps and the matched 2-ms references. Colour saturation in heatmaps does not alter printed values.','', '## Scope and interpretation','', 'This extends the P1 timing question to within-participant decoding across P1–P12. It does not train a pooled population decoder or test an unseen participant. Training uses series 1–6, validation 7, test 8–9. Test series are weight series at surface code 3 for all participants; generalization to all task conditions is not established.','', 'The end gap changes the EEG history interval, without establishing an optimal physiological delay. Inputs are 128 samples, stride64, at500Hz. Larger gaps can span different movement stages. No phase-specific population claim follows from this report. Training-only channel and target scalers differ by participant. MSE values therefore use each participant’s training-standardized wrist coordinates.','', 'P1 tests and prior transfer results informed planning. The data are exploratory, with a single validation series and partly undocumented upstream preprocessing. The bounded ridge solver and final15-epoch CNN budget remain fixed. No model/seed/gap is retuned from test results. Longer input histories and temporal bins belong to the separately executed EXP-018 / T2 and are outside this fixed-window T1 comparison.','', '## Numerical evidence','', '- `results.json`: every candidate validation metric, selected alpha, seed history, scaler and permitted test score.','- `selection.json`: global frozen validation choices.','- `population_summary.json`: participant-first metrics and participant-bootstrap intervals.','- `validation_candidates.csv`, `participant_metrics.csv`, `seed_comparisons.csv`: inspectable tables.','- Participant subdirectories retain local ignored weights, validation predictions, test predictions and common split indices.','- `verification.json`: no-fitting independent input/index/scaler/selection and saved-weight replays.','', '[Population protocol](../../documentation/methods/POPULATION_TIMING_ALIGNMENT.md) · [Registry](../../documentation/EXPERIMENT_REGISTRY.md)']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'report_source.py').write_bytes(Path(__file__).read_bytes());(out/'summary_helper_source.py').write_bytes((ROOT/'src/eeg_kinematics/population_timing.py').read_bytes())
    print(json.dumps(s['population'],indent=2))

if __name__=='__main__':main()
