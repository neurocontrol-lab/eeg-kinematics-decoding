"""Fixed retrospective feature and weighting definitions for EXP-036."""
import numpy as np
from scipy.signal import savgol_filter

STAGES=['unassigned/intertrial','preparation','reach','preload','loading','airborne_before_lowering_cue','after_lowering_cue','release','return']
MODELS=['training_mean','ridge','cnn','cnn_bilstm','cnn_elu','eegnet_regression']
WINDOWS=[21,41,81]
SPECS={'constant':'','position':'P','stage':'S','dynamics':'D','position_stage':'PS','position_dynamics':'PD','stage_dynamics':'SD','combined':'PSD','combined_jerk':'PSDJ'}

def derivatives(positions,window,fs=500):
    """Caller passes one complete recording series, never concatenated series."""
    return np.stack([savgol_filter(positions,window,3,deriv=d,delta=1/fs,axis=0,mode='interp') for d in [1,2,3]],axis=1)

def quadratic(x):
    """Linear terms, then x_i*x_j for i <= j, with a fixed column order."""
    return np.column_stack([x]+[x[:,i]*x[:,j] for i in range(x.shape[1]) for j in range(i,x.shape[1])])

def groups(position,stage,deriv):
    v,a,j=deriv[:,0],deriv[:,1],deriv[:,2]
    return {'P':quadratic(position),'S':np.eye(9)[stage][:,1:],
            'D':quadratic(np.column_stack([v,a,np.linalg.norm(v,axis=1),np.linalg.norm(a,axis=1)])),
            'J':quadratic(np.column_stack([j,np.linalg.norm(j,axis=1)]))}

def trial_weights(keys):
    _,inv,counts=np.unique(keys,return_inverse=True,return_counts=True)
    w=1.0/counts[inv]
    return w/w.mean()

def scores(y,p,keys):
    w=trial_weights(keys);err=p-y
    mse=float(np.average(err**2,weights=w));variance=float(np.average((y-np.average(y,weights=w))**2,weights=w))
    return {'n':len(y),'trials':len(np.unique(keys)),'mse':mse,'mae':float(np.average(np.abs(err),weights=w)),
            'r2_error_target':1-mse/variance if variance>0 else None,'negative_prediction_fraction':float(np.mean(p<0))}

def block_interval(delta,keys,block,reps,seed=42):
    rng=np.random.default_rng(seed);byrun=[delta[(keys//100)==r] for r in [8,9]]
    values=[]
    for _ in range(reps):
        sample=[]
        for a in byrun:
            starts=rng.integers(0,len(a),size=int(np.ceil(len(a)/block)))
            idx=((starts[:,None]+np.arange(block))%len(a)).ravel()[:len(a)]
            sample.extend(a[idx])
        values.append(np.mean(sample))
    return {'block_trials':block,'replicates':reps,'seed':seed,'mean':float(np.mean(delta)),
            'lower_95':float(np.quantile(values,.025)),'upper_95':float(np.quantile(values,.975))}
