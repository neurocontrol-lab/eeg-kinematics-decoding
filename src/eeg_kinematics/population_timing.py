"""Participant-first summaries for the predeclared population timing comparison."""
import numpy as np

def summarize_population(result):
    """Average repeated-fit metrics within participant before population inference."""
    assert result['status']=='complete' and result['experiment_id']=='EXP-042'
    people=result['participants'];seeds=result['cnn_seeds'];rows=[]
    draw=np.random.default_rng(20261028).integers(0,len(people),size=(10000,len(people)))
    for person in people:
        tests=result['participant_results'][person]['test_evaluations']
        for model in ['ridge','cnn_elu']:
            gap=result['selection'][person][model]['gap_samples']
            keys=lambda g:([f'ridge_gap{g}'] if model=='ridge' else [f'cnn_elu_gap{g}_seed{s}' for s in seeds])
            b=[tests[k]['aggregate'] for k in keys(1)];s=[tests[k]['aggregate'] for k in keys(gap)]
            baseline=float(np.mean([v['mse_mean'] for v in b]));selected=float(np.mean([v['mse_mean'] for v in s]));assert baseline>0
            row={'participant':person,'model':model,'selected_gap_ms':gap*2,'test_targets':b[0]['n'],'baseline_mse':baseline,'selected_mse':selected,'selected_minus_baseline':selected-baseline,'percent_mse_reduction':100*(baseline-selected)/baseline,'seed_mse_differences':[y['mse_mean']-x['mse_mean'] for x,y in zip(b,s)],'seed_percent_reductions':[100*(x['mse_mean']-y['mse_mean'])/x['mse_mean'] for x,y in zip(b,s)]}
            for name,values in [('baseline',b),('selected',s)]:
                for metric in ['mse','rmse','mae','r2']:
                    row[name+'_axis_'+metric]=[float(np.mean([v[metric][a] for v in values])) if all(v[metric][a] is not None for v in values) else None for a in range(3)]
            row['by_series']={}
            for series in ['8','9']:
                x=float(np.mean([tests[k]['by_series'][series]['mse_mean'] for k in keys(1)]))
                y=float(np.mean([tests[k]['by_series'][series]['mse_mean'] for k in keys(gap)]))
                row['by_series'][series]={'baseline_mse':x,'selected_mse':y,'selected_minus_baseline':y-x}
            rows.append(row)
    population={}
    for model in ['ridge','cnn_elu']:
        rr=[r for r in rows if r['model']==model];delta=np.array([r['selected_minus_baseline'] for r in rr]);percent=np.array([r['percent_mse_reduction'] for r in rr])
        population[model]={'participants':len(rr),'mean_baseline_mse':float(np.mean([r['baseline_mse'] for r in rr])),'mean_selected_mse':float(np.mean([r['selected_mse'] for r in rr])),'mean_selected_minus_baseline':float(delta.mean()),'paired_mse_difference_95_interval':np.percentile(delta[draw].mean(axis=1),[2.5,97.5]).tolist(),'mean_participant_percent_reduction':float(percent.mean()),'percent_reduction_95_interval':np.percentile(percent[draw].mean(axis=1),[2.5,97.5]).tolist(),'median_participant_percent_reduction':float(np.median(percent)),'participants_improved':int((delta<0).sum()),'participants_unchanged':int((delta==0).sum()),'participants_worse':int((delta>0).sum()),'selected_gap_counts':{str(g):sum(r['selected_gap_ms']==g for r in rr) for g in result['gap_ms']}}
    return {'experiment_id':'EXP-042','participant_count':len(people),'aggregation':'Mean repeated-fit metrics within each participant, then equal participant weight. No prediction ensemble.','primary_contrast':'Selected minus 2-ms standardized XYZ MSE. Negative favours selected.','bootstrap':{'unit':'participant','replicates':10000,'seed':20261028,'scope':'Exploratory percentile intervals across the dataset participants, conditional on fitted seeds and the fixed series split.'},'participant_metrics':rows,'population':population}
