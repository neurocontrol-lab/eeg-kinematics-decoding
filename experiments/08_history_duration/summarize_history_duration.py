"""Summarize fixed-history and validation-selected scores with equal participant weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output_dir',type=Path);args=ap.parse_args()
    out=args.output_dir.resolve();d=json.loads((out/'results.json').read_text())
    assert d['status']=='complete' and len(d['participants'])==12
    frozen=json.loads((out/'selection.json').read_text());rows=[];selected={}
    people=[d['participants'][f'P{i}'] for i in range(1,13)]
    for model in ('ridge','cnn_elu'):
        for h in (128,256,512):
            scores=[p['test'][f'{model}_{h}']['aggregate'] for p in people]
            mse=np.array([s['mse_mean'] for s in scores]);r2=np.array([s['r2'] for s in scores])
            rows.append({'model':model,'history_ms':h*2,'participants':12,'macro_mse':float(mse.mean()),
                         'median_participant_mse':float(np.median(mse)),
                         'macro_r2_x':float(r2[:,0].mean()),'macro_r2_y':float(r2[:,1].mean()),
                         'macro_r2_z':float(r2[:,2].mean()),'macro_mean_axis_r2':float(r2.mean())})
        chosen=[frozen['models'][f'P{i}'][model] for i in range(1,13)]
        scores=[p['test'][f'{model}_{h}']['aggregate'] for p,h in zip(people,chosen)]
        base=np.array([p['test'][f'{model}_128']['aggregate']['mse_mean'] for p in people])
        mse=np.array([s['mse_mean'] for s in scores])
        selected[model]={'history_ms_counts':{str(h*2):chosen.count(h) for h in (128,256,512)},
                         'macro_mse':float(mse.mean()),'macro_r2_axes':np.array([s['r2'] for s in scores]).mean(axis=0).tolist(),
                         'mean_relative_reduction_pct':float(np.mean(100*(base-mse)/base)),
                         'improved_vs_reference':int((mse<base).sum()),'unchanged_vs_reference':int((mse==base).sum())}
    budgets={}
    for h in (128,256,512):
        all_iterations=[i for p in people for c in p['histories'][str(h)]['ridge']['candidates'] for i in c['n_iter']]
        chosen_iterations=[i for p in people for i in p['histories'][str(h)]['ridge']['n_iter']]
        budgets[str(h*2)]={'candidate_target_solves':len(all_iterations),'candidate_solves_at_200':sum(i==200 for i in all_iterations),
                           'selected_target_solves':len(chosen_iterations),'selected_solves_at_200':sum(i==200 for i in chosen_iterations)}
    summary={'experiment_id':'EXP-018','weighting':'Each of 12 separately trained participants has weight 1/12; no pooling of windows.',
             'fixed_histories':rows,'validation_selected_histories':selected,'paired_effects':d['population'],
             'ridge_iteration_budget_diagnostics':budgets}
    (out/'group_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'group_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    keys=list(d['population']);fig,ax=plt.subplots(figsize=(9,4.5))
    for i,key in enumerate(keys):
        e=d['population'][key];mean=e['mean_relative_mse_reduction_pct'];lo,hi=e['mean_relative_reduction_bootstrap_95']
        ax.errorbar(mean,i,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o',capsize=4,color='#147c87')
        ax.annotate(f"{e['improved']}/12 improved",(hi,i),xytext=(8,0),textcoords='offset points',va='center',fontsize=9)
    ax.axvline(0,color='black',linewidth=.8);ax.set_yticks(range(len(keys)),[k.replace('cnn_elu','CNN + ELU').replace('ridge','Ridge').replace('_',' / ') for k in keys]);ax.invert_yaxis()
    ax.set(xlabel='Mean participant MSE reduction (%) vs 256 ms',title='T2: paired effects across 12 participant-specific decoders')
    ax.grid(axis='x',alpha=.2);ax.margins(x=.3);fig.tight_layout();fig.savefig(out/'group_effects.png',dpi=180);plt.close(fig)
    source=Path(__file__).read_bytes();(out/'summary_source.py').write_bytes(source)
    receipt={'source_sha256':hashlib.sha256(source).hexdigest(),'results_sha256':hashlib.sha256((out/'results.json').read_bytes()).hexdigest(),
             'artifacts':{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('group_summary.json','group_metrics.csv','group_effects.png')}}
    (out/'summary_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
