"""Matched target indices and channel-wise temporal mean bins for EXP-018."""
import numpy as np

BINS = 128
HISTORIES = (128, 256, 512)
STRIDE = 64


def matched_targets(bounds, runs):
    """Keep the original run-relative grid, eligible for every declared history."""
    targets, ids = [], []
    for run in runs:
        lo, hi = bounds[run]
        grid = np.arange(lo + BINS, hi, STRIDE, dtype=np.int64)
        grid = grid[grid - max(HISTORIES) >= lo]
        targets.extend(grid)
        ids.extend([run] * len(grid))
    return np.asarray(targets, dtype=np.int64), np.asarray(ids, dtype=np.int8)


def mean_bin_windows(eeg, targets, ids, bounds, history):
    """Last input is t-1; consecutive blocks of 1/2/4 samples make 128 bins."""
    if history not in HISTORIES:
        raise ValueError(history)
    targets = np.asarray(targets, dtype=np.int64)
    if len(targets) != len(ids):
        raise ValueError('Target/run counts differ')
    for run in np.unique(ids):
        t = targets[np.asarray(ids) == run]
        lo, hi = bounds[int(run)]
        if not (np.all(t - history >= lo) and np.all(t < hi)):
            raise ValueError('History or target crosses a recording-series boundary')
    result = np.empty((len(targets), BINS, eeg.shape[1]), dtype=np.float32)
    for start in range(0, len(targets), 256):
        batch = targets[start:start + 256]
        raw = np.stack([eeg[t-history:t] for t in batch])
        result[start:start+len(batch)] = raw.reshape(len(batch), BINS, history//BINS, eeg.shape[1]).mean(axis=2)
    return result
