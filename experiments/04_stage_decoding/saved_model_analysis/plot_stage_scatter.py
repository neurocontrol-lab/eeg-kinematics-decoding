"""Fit stage-colored scatter panels to the data, preserving equal x/y scale."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import argparse,csv,hashlib,json
from pathlib import Path
import sys
_repo = next(p for p in Path(__file__).resolve().parents if (p / "src/eeg_kinematics/paths.py").is_file())
sys.path.insert(0, str(_repo / "src"))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_stage_scatter(y,pred,phases,stages,output):
    assert y.shape==pred.shape and y.shape[1]==3
    assert np.isfinite(y).all() and np.isfinite(pred).all()
    colors=plt.cm.Set3(np.linspace(0,1,len(stages)))
    labels=['Unassigned/intertrial','Preparation','Reach','Preload','Loading','Airborne before cue','After lowering cue','Release','Return']
    fig,axes=plt.subplots(1,3,figsize=(14,5.8))
    fig.subplots_adjust(left=.065,right=.99,top=.83,bottom=.29,wspace=.28)
    settings={}
    for a,ax in enumerate(axes):
        lo=float(min(y[:,a].min(),pred[:,a].min()));hi=float(max(y[:,a].max(),pred[:,a].max()))
        pad=max((hi-lo)*.04,1e-4);lo-=pad;hi+=pad
        for si,stage in enumerate(stages):
            mask=phases==si
            ax.scatter(y[mask,a],pred[mask,a],s=7,alpha=.4,color=colors[si],label=labels[si])
        ax.plot([lo,hi],[lo,hi],'k--',linewidth=.8,zorder=0)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),title='Wrist '+'XYZ'[a],xlabel='Observed (reduced)',ylabel='Predicted (reduced)')
        ax.set_aspect('equal',adjustable='box')
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(alpha=.12)
        settings['XYZ'[a]]={'x_limits':[lo,hi],'y_limits':[lo,hi],'equal_aspect':True,'points':len(y)}
        assert np.all((y[:,a]>lo)&(y[:,a]<hi)) and np.all((pred[:,a]>lo)&(pred[:,a]<hi))
    handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.5,.025),ncol=3,fontsize=9,markerscale=1.5)
    fig.suptitle('CNN–BiLSTM scatter points traced to target event phase',y=.97)
    fig.text(.5,.90,'Data-fitted panels; observed/predicted axes share limits and equal scale within each panel.',ha='center',fontsize=10)
    fig.savefig(output,dpi=160);plt.close(fig)
    return settings

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('analysis_dir',type=Path)
    out=parser.parse_args().analysis_dir.resolve();source=out/'aligned_predictions.csv'
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    rows=[r for r in csv.DictReader(source.open()) if r['model']=='cnn_bilstm']
    stages=list(json.loads((out/'results.json').read_text())['models']['cnn_bilstm']['by_stage'])
    y=np.array([[float(r['y_'+a]) for a in 'xyz'] for r in rows]);pred=np.array([[float(r['pred_'+a]) for a in 'xyz'] for r in rows])
    phases=np.array([stages.index(r['stage']) for r in rows])
    settings=plot_stage_scatter(y,pred,phases,stages,out/'scatter_by_stage.png')
    assert before==hashlib.sha256(source.read_bytes()).hexdigest()
    (out/'scatter_plot_settings.json').write_text(json.dumps({'panels':settings,'source_sha256':before,'all_points_included':True},indent=2)+'\n')
    print(json.dumps(settings,indent=2))

if __name__=='__main__':main()
