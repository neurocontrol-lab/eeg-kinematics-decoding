"""Test physical bin coverage, matched target grids and series boundaries."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from eeg_kinematics.history_duration import matched_targets, mean_bin_windows, HISTORIES


class HistoryDurationTests(unittest.TestCase):
    def test_shared_grid_and_exact_bin_means(self):
        bounds = {1: (0, 1025), 2: (1025, 2050)}
        t, ids = matched_targets(bounds, [1, 2])
        np.testing.assert_array_equal(t[ids == 1], np.arange(512, 1025, 64))
        np.testing.assert_array_equal(t[ids == 2], np.arange(1537, 2050, 64))
        eeg = np.column_stack([np.arange(2050), -np.arange(2050)]).astype(np.float32)
        for history in HISTORIES:
            windows = mean_bin_windows(eeg, t, ids, bounds, history)
            self.assertEqual(windows.shape, (len(t), 128, 2))
            for i, target in enumerate(t):
                expected = np.array([np.mean(eeg[a:a+history//128], axis=0)
                                     for a in range(target-history, target, history//128)])
                np.testing.assert_array_equal(windows[i], expected)

    def test_no_future_or_neighbouring_series(self):
        eeg = np.zeros((1200, 2), np.float32)
        bounds = {1: (0, 600), 2: (600, 1200)}
        with self.assertRaises(ValueError):
            mean_bin_windows(eeg, np.array([700]), np.array([2]), bounds, 512)
        eeg[1100:] = 999
        w = mean_bin_windows(eeg, np.array([1100]), np.array([2]), bounds, 256)
        self.assertTrue(np.all(w == 0))


if __name__ == '__main__':
    unittest.main()
