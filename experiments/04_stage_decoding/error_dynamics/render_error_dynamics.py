"""Re-render EXP-036 summaries from saved numbers; no fitting or inference."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output_dir',type=Path);args=ap.parse_args();out=args.output_dir
    d=json.loads((out/'results.json').read_text());bins=list(csv.DictReader((out/'speed_bins.csv').open()));models=d['models']
    stages=['preparation','reach','preload','loading','airborne_before_lowering_cue','after_lowering_cue','release','return']
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    for st,ax in zip(stages,axes.flat):
        rr=[r for r in bins if int(r['window_samples'])==41 and r['decoder']=='cnn_bilstm' and r['stage']==st and int(r['trials'])>=5]
        title=st.replace('_',' ')
        if st=='airborne_before_lowering_cue':title='Airborne before\nlowering cue'
        if rr:
            x=np.array([float(r['median_speed']) for r in rr]);y=np.array([float(r['trial_mean_error']) for r in rr]);ax.plot(x,y,'o-',lw=1.5);ax.vlines(x,[float(r['trial_q25']) for r in rr],[float(r['trial_q75']) for r in rr],alpha=.5,lw=2)
            ax.set_xscale('log');ax.xaxis.set_major_locator(FixedLocator(np.geomspace(x.min(),x.max(),3)));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:.2g}'));ax.xaxis.set_minor_locator(NullLocator())
        ax.set(title=title,xlabel='Speed (reduced units/s)',ylabel='Trial-mean error')
    fig.suptitle('Within-stage speed curves for CNN–BiLSTM\nEqual contributing-trial means, IQR bars; at least 5 trials/bin. Axis ranges differ.',fontsize=14)
    fig.savefig(out/'within_stage_speed.png',dpi=160,bbox_inches='tight');plt.close(fig)
    specs=list(d['feature_specs']);mat=np.array([[d['scores']['41'][m][s]['combined']['r2_error_target'] for s in specs] for m in models])
    fig,ax=plt.subplots(figsize=(14,5),layout='constrained');im=ax.imshow(mat,cmap='RdYlGn',vmin=-.5,vmax=1.,aspect='auto');ax.set_xticks(np.arange(9),['Constant','Position','Stage','Dynamics','P + S','P + D','S + D','P + S + D','P + S + D + J'],rotation=25,ha='right');ax.set_yticks(np.arange(6),models)
    for a in range(6):
        for b in range(9):ax.text(b,a,f'{mat[a,b]:.2f}',ha='center',va='center',fontsize=10)
    ax.set_title('Explaining decoder error across series (trial-balanced, 82-ms smoothing)\nThese R² values describe prediction error, not wrist reconstruction');fig.colorbar(im,ax=ax,label='Out-of-series error-target R²');fig.savefig(out/'explanatory_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),gridspec_kw={'width_ratios':[1,4]},layout='constrained')
    for ax,indices in zip(axes,[[0],list(range(1,6))]):
        for window,offset in [(21,-.2),(41,0),(81,.2)]:
            values=[]
            for i in indices:
                s=d['scores'][str(window)][models[i]];a=s['position_stage']['combined']['mse'];b=s['combined']['combined']['mse'];values.append(100*(a-b)/a)
            ax.bar(np.arange(len(indices))+offset,values,.19,label=f'{window*2}-ms smoothing')
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(np.arange(len(indices)),[models[i] for i in indices],rotation=15);ax.set_ylabel('Explanatory MSE reduction (%)')
    axes[0].set_title('Constant wrist reference');axes[1].set_title('Learned wrist decoders');axes[1].legend()
    fig.suptitle('Adding dynamics beyond position and stage\nPositive favours combined features; separate y-axis scales',fontsize=14);fig.savefig(out/'incremental_dynamics.png',dpi=160,bbox_inches='tight');plt.close(fig)
    region=list(csv.DictReader((out/'scatter_regions.csv').open()));fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    specs=['position_stage','combined','combined_jerk'];labels=['Position + stage','Add dynamics','Add jerk'];values=[d['scores']['41']['cnn_bilstm'][s]['combined']['mse'] for s in specs]
    bars=axes[0].bar(np.arange(3),values,color=['#557A95','#B65D22','#888888']);axes[0].bar_label(bars,fmt='%.5f',padding=5);axes[0].set_xticks(np.arange(3),labels);axes[0].set_ylim(0,max(values)*1.22);axes[0].set_ylabel('Error-explanation MSE (lower is better)');axes[0].set_title('CNN–BiLSTM: cross-series error explanation')
    ci=d['primary_contrast']['intervals'][0];axes[0].text(.03,.94,f'Adding dynamics: +1.20% MSE\n34/68 trials improve; paired interval spans zero',transform=axes[0].transAxes,va='top',fontsize=10)
    for j,(code,label,color) in enumerate([('near_diagonal','Near diagonal','#228B68'),('observed_anchor_only_high_error','Observed-anchor streak','#D95F02'),('predicted_anchor_only_high_error','Predicted-anchor streak','#756BB1')]):
        values=[float(next(r['median_speed'] for r in region if r['decoder']=='cnn_bilstm' and r['axis']==a and r['region']==code)) for a in 'XYZ']
        axes[1].bar(np.arange(3)+(j-1)*.25,values,.24,label=label,color=color)
    axes[1].set_xticks(np.arange(3),list('XYZ'));axes[1].set_yscale('log');axes[1].set_ylabel('Median observed speed (reduced units/s)');axes[1].set_title('Scatter streaks differ across axes');axes[1].legend(fontsize=9)
    fig.suptitle('R6: dynamics describe scatter structure, with no clear added benefit for the lead error model\nP1, two familiar series, fixed saved decoders. Error-explanation scores do not improve wrist decoding.',fontsize=12)
    fig.savefig(out/'meeting_summary.png',dpi=180,bbox_inches='tight');plt.close(fig)
    (out/'render_source.py').write_bytes(Path(__file__).read_bytes())
    receipt={'operation':'plot-only layout repair and summary figure from verified saved summaries','source_sha256':hashlib.sha256((out/'render_source.py').read_bytes()).hexdigest(),'replaced_plots':['within_stage_speed.png','explanatory_comparison.png','incremental_dynamics.png'],'new_summary_plot':'meeting_summary.png','models_fitted':0}
    (out/'plot_rendering.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Re-rendered three summaries without model fitting.')

if __name__=='__main__':main()
