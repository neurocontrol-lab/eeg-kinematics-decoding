"""Guard against seed/window pseudoreplication in population summaries."""
import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from eeg_kinematics.population_timing import summarize_population

def metric(value,n):
    return {'n':n,'mse_mean':value,'mse':[value]*3,'rmse':[value**.5]*3,'mae':[value**.5]*3,'r2':[.1]*3}

def fixture():
    d={'experiment_id':'EXP-042','status':'complete','participants':['P1','P2'],'cnn_seeds':[42,43,44],'gap_ms':[2,50],'selection':{},'participant_results':{}}
    for person,n,b,s in [('P1',10,[1,2,10],[.9,1,4]),('P2',1000,[4,4,4],[4,4,4])]:
        d['selection'][person]={'ridge':{'gap_samples':1},'cnn_elu':{'gap_samples':25}}
        tests={}
        for seed,x,y in zip([42,43,44],b,s):
            for gap,value in [(1,x),(25,y)]:
                m=metric(value,n);tests[f'cnn_elu_gap{gap}_seed{seed}']={'aggregate':m,'by_series':{'8':m,'9':m}}
        m=metric(2,n);tests['ridge_gap1']={'aggregate':m,'by_series':{'8':m,'9':m}}
        d['participant_results'][person]={'test_evaluations':tests}
    return d

class PopulationTests(unittest.TestCase):
    def test_equal_participant_weight_after_seed_average(self):
        s=summarize_population(fixture());r=[r for r in s['participant_metrics'] if r['participant']=='P1' and r['model']=='cnn_elu'][0]
        self.assertAlmostEqual(r['baseline_mse'],13/3)
        self.assertAlmostEqual(r['selected_mse'],5.9/3)
        self.assertAlmostEqual(r['percent_mse_reduction'],100*(13-5.9)/13)
        p=s['population']['cnn_elu']
        self.assertAlmostEqual(p['mean_selected_minus_baseline'],(5.9-13)/6)
        self.assertEqual((p['participants_improved'],p['participants_unchanged'],p['participants_worse']),(1,1,0))

    def test_reference_selected_means_exact_zero_and_fixed_bootstrap(self):
        s=summarize_population(fixture())
        self.assertEqual(s['population']['ridge']['paired_mse_difference_95_interval'],[0,0])
        self.assertEqual(s['population']['ridge']['participants_unchanged'],2)
        d=np.array([(5.9-13)/3,0])
        ix=np.random.default_rng(20261028).integers(0,2,size=(10000,2))
        np.testing.assert_allclose(np.percentile(d[ix].mean(axis=1),[2.5,97.5]),s['population']['cnn_elu']['paired_mse_difference_95_interval'],rtol=1e-14,atol=1e-14)

if __name__=='__main__':unittest.main()
