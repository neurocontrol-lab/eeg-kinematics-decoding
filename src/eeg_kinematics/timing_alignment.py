"""Indexing for matched-target EEG timing comparisons (no model fitting)."""
import numpy as np

WINDOW=128
STRIDE=64
GAPS=(1,25,50,100)  # Samples from final EEG input to target: 2/50/100/200 ms.

def matched_targets(bounds,runs,gaps=GAPS):
    targets=[];ids=[]
    for run in runs:
        lo,hi=bounds[run]
        grid=np.arange(lo+WINDOW,hi,STRIDE,dtype=np.int64)
        grid=grid[grid-(WINDOW-1)-max(gaps)>=lo]
        targets.extend(grid);ids.extend([run]*len(grid))
    return np.asarray(targets,dtype=np.int64),np.asarray(ids,dtype=np.int8)

def window_starts(targets,gap):
    return np.asarray(targets,dtype=np.int64)-int(gap)-(WINDOW-1)

def make_windows(eeg,targets,ids,bounds,gap):
    starts=window_starts(targets,gap)
    for run in np.unique(ids):
        mask=ids==run;lo,hi=bounds[int(run)]
        assert np.all(starts[mask]>=lo)
        assert np.all(targets[mask]<hi)
        assert np.all(starts[mask]+WINDOW-1==targets[mask]-gap)
    return np.stack([eeg[s:s+WINDOW] for s in starts]).astype(np.float32,copy=False)
