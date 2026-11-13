"""Recompute participant-first summaries from saved predictions, without fitting."""
import argparse,csv,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from sklearn.metrics import r2_score

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output_dir',type=Path);a=ap.parse_args();out=a.output_dir.resolve()
    d=json.loads((out/'results.json').read_text());s=json.loads((out/'population_summary.json').read_text());v=json.loads((out/'verification.json').read_text())
    assert d['experiment_id']==s['experiment_id']==v['experiment_id']=='EXP-042' and d['status']=='complete' and v['status']=='passed'
    assert s['participant_count']==12 and len(s['participant_metrics'])==24
    comparisons={};arrays=0
    for person in d['participants']:
        for model in ['ridge','cnn_elu']:
            g=d['selection'][person][model]['gap_samples'];collected={}
            for label,gap in [('baseline',1),('selected',g)]:
                metrics=[];series_metrics={'8':[],'9':[]}
                for seed in ([None] if model=='ridge' else [42,43,44]):
                    key=f'{model}_gap{gap}'+('' if seed is None else f'_seed{seed}')
                    with np.load(out/person/(key+'_predictions.npz')) as z:y=z['y_true'].astype(float);p=z['y_pred'].astype(float);ids=z['series_id']
                    errors=p-y;ms=np.mean(errors**2,axis=0)
                    metrics.append({'mse':ms,'mean':ms.mean(),'rmse':np.sqrt(ms),'mae':np.abs(errors).mean(axis=0),'r2':r2_score(y,p,multioutput='raw_values')})
                    for r in ['8','9']:series_metrics[r].append(float(np.mean(errors[ids==int(r)]**2)))
                    arrays+=1
                collected[label]=(metrics,series_metrics)
            row=next(r for r in s['participant_metrics'] if r['participant']==person and r['model']==model)
            b,selected=collected['baseline'][0],collected['selected'][0]
            bm=np.mean([m['mean'] for m in b]);sm=np.mean([m['mean'] for m in selected]);delta=sm-bm;pct=100*(bm-sm)/bm
            for name,expected in [('baseline_mse',bm),('selected_mse',sm),('selected_minus_baseline',delta),('percent_mse_reduction',pct)]:np.testing.assert_allclose(row[name],expected,rtol=1e-10,atol=1e-12)
            for label in ['baseline','selected']:
                for metric in ['mse','rmse','mae','r2']:np.testing.assert_allclose(row[label+'_axis_'+metric],np.mean([m[metric] for m in collected[label][0]],axis=0),rtol=1e-9,atol=1e-11)
                for r in ['8','9']:np.testing.assert_allclose(row['by_series'][r][label+'_mse'],np.mean(collected[label][1][r]),rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(row['seed_mse_differences'],[y['mean']-x['mean'] for x,y in zip(b,selected)],rtol=1e-10,atol=1e-12)
            assert row['selected_gap_ms']==g*2 and row['test_targets']==len(y)
            comparisons[(person,model)]=(bm,sm,delta,pct)
    draws=np.random.default_rng(20261028).integers(0,12,size=(10000,12))
    for model in ['ridge','cnn_elu']:
        table=np.array([comparisons[(p,model)] for p in d['participants']]);q=s['population'][model]
        for name,expected in [('mean_baseline_mse',table[:,0].mean()),('mean_selected_mse',table[:,1].mean()),('mean_selected_minus_baseline',table[:,2].mean()),('mean_participant_percent_reduction',table[:,3].mean()),('median_participant_percent_reduction',np.median(table[:,3]))]:np.testing.assert_allclose(q[name],expected,rtol=1e-10,atol=1e-12)
        for name,column in [('paired_mse_difference_95_interval',2),('percent_reduction_95_interval',3)]:np.testing.assert_allclose(q[name],np.percentile(table[draws,column].mean(axis=1),[2.5,97.5]),rtol=1e-10,atol=1e-12)
        assert q['participants_improved']==int((table[:,2]<0).sum()) and q['participants_unchanged']==int((table[:,2]==0).sum()) and q['participants_worse']==int((table[:,2]>0).sum())
        assert sum(q['selected_gap_counts'].values())==12
    with (out/'participant_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==24
    for row in rows:
        b,selected,delta,pct=comparisons[(row['participant'],row['model'])]
        np.testing.assert_allclose([float(row[n]) for n in ['baseline_mse','selected_mse','selected_minus_baseline','percent_mse_reduction']],[b,selected,delta,pct],rtol=1e-10,atol=1e-12)
    with (out/'validation_candidates.csv').open() as f:assert len(list(csv.DictReader(f)))==576
    with (out/'seed_comparisons.csv').open() as f:assert len(list(csv.DictReader(f)))==48
    for name in ['meeting_summary.png','validation_heatmaps.png','validation_curves.png','paired_test_scores.png','participant_improvements.png','selected_gap_counts.png','axis_test_r2.png','RESULTS.md']:assert (out/name).stat().st_size>0
    result={'experiment_id':'EXP-042','status':'passed','checked_at':datetime.now(timezone.utc).isoformat(),'participant_model_comparisons':24,'saved_prediction_array_reads':arrays,'population_bootstrap_intervals_recomputed':4,'candidate_csv_rows':576,'seed_csv_rows':48,'participant_csv_rows':24,'models_fitted':0,'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'summary_verifier_source.py').write_bytes(Path(__file__).read_bytes());(out/'population_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
