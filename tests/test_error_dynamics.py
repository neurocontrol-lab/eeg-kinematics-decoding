"""Scientific invariants for retrospective derivative and weighting definitions."""
import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from eeg_kinematics.error_dynamics import derivatives,trial_weights

class DynamicsTests(unittest.TestCase):
    def test_cubic_trajectory_recovers_all_three_derivatives(self):
        t=np.arange(501)/500;x=np.column_stack([t**3,2*t**3,-t**3])
        for window in [21,41,81]:
            actual=derivatives(x,window)[40:-40];tt=t[40:-40]
            for a,scale in enumerate([1,2,-1]):
                np.testing.assert_allclose(actual[:,0,a],scale*3*tt**2,atol=1e-9)
                np.testing.assert_allclose(actual[:,1,a],scale*6*tt,atol=1e-8)
                np.testing.assert_allclose(actual[:,2,a],scale*6,atol=1e-6)

    def test_separate_series_do_not_create_boundary_motion(self):
        for offset in [0,1000]:
            d=derivatives(np.full((201,3),float(offset)),41)
            np.testing.assert_allclose(d[40:-40],0,atol=1e-5)

    def test_each_trial_has_equal_total_weight(self):
        keys=np.array([801]*2+[802]*7+[901]*3);w=trial_weights(keys)
        totals=[w[keys==k].sum() for k in np.unique(keys)]
        np.testing.assert_allclose(totals,[4,4,4]);self.assertAlmostEqual(w.mean(),1)

if __name__=='__main__':unittest.main()
