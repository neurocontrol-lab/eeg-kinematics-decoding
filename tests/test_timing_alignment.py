"""Check matched-target gaps and run isolation on synthetic labelled signals."""
import unittest,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from eeg_kinematics.timing_alignment import matched_targets,make_windows,GAPS

class TimingTests(unittest.TestCase):
    def test_earliest_common_grid_point_and_exact_input_end(self):
        bounds={1:(0,512),2:(512,1024)}
        x=np.arange(1024,dtype=np.float32)[:,None]
        target,ids=matched_targets(bounds,[1,2])
        np.testing.assert_array_equal(target,[256,320,384,448,768,832,896,960])
        for gap in GAPS:
            w=make_windows(x,target,ids,bounds,gap)
            np.testing.assert_array_equal(w[:,-1,0],target-gap)
            np.testing.assert_array_equal(w[:,0,0],target-gap-127)
            self.assertEqual(w.shape,(8,128,1))

    def test_run_offset_does_not_shift_stride_origin(self):
        bounds={1:(0,513),2:(513,1030)}
        t,ids=matched_targets(bounds,[2])
        np.testing.assert_array_equal(t,[769,833,897,961,1025])
        self.assertTrue(np.all(ids==2))
        with self.assertRaises(AssertionError):make_windows(np.zeros((1030,1)),np.array([640]),np.array([2]),bounds,100)

if __name__=='__main__':unittest.main()
