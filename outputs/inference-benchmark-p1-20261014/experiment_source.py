"""Warmed host-input/host-output latency for the six saved P1 references.

Separate CPU/GPU processes, fixed four CPU threads, synchronized NumPy output.
This measures model calls, excluding acquisition, scaling, windowing and loading.
"""
import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--device', choices=['cpu', 'gpu'], required=True)
args = parser.parse_args()
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS'):
    os.environ[key] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
if args.device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import csv
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
import numpy as np
import scipy
from scipy.io import loadmat
import tensorflow as tf

NAMES = ('training_mean', 'ridge', 'cnn', 'cnn_bilstm', 'cnn_elu', 'eegnet_regression')
source = args.repo / 'outputs/six-model-comparison-p1-gpu-20260930T095506Z'
args.output.mkdir(parents=True, exist_ok=True)
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
artifacts = [source/'results.json', source/'training_mean_weights.npz', source/'ridge_weights.npz']
artifacts += [source/f'{n}.keras' for n in NAMES[2:]]
hashes = {p.name: digest(p) for p in artifacts}
metadata = json.loads((source/'results.json').read_text())
with np.load(source/'cnn_predictions.npz') as saved:
    indices = saved['target_sample_index'].copy()
eeg = np.asarray(loadmat(args.repo/'way_eeg_clean/eeg_P1.mat', variable_names=['eeg'])['eeg'].T, dtype=np.float32, order='C')
eeg -= np.asarray(metadata['scaling']['eeg_mean'])
eeg /= np.asarray(metadata['scaling']['eeg_scale'])
x = np.stack([eeg[int(t)-128:int(t)] for t in indices])
assert x.shape == (3642, 128, 32) and np.isfinite(x).all()
del eeg
gpu = tf.config.list_physical_devices('GPU')
device = '/GPU:0' if args.device == 'gpu' else '/CPU:0'
if args.device == 'gpu':
    assert gpu, 'GPU required, no fallback'
    for g in gpu:
        tf.config.experimental.set_memory_growth(g, True)
with tf.device(device):
    probe = tf.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
assert device.lstrip('/').lower() in probe.device.lower()
probe.numpy()
models = {}
initial = {}
for n in NAMES[2:]:
    with tf.device(device):
        models[n] = tf.keras.models.load_model(source/f'{n}.keras', compile=False)
    initial[n] = [w.copy() for w in models[n].get_weights()]
with np.load(source/'training_mean_weights.npz') as f:
    mean = f['mean'].copy()
with np.load(source/'ridge_weights.npz') as f:
    coef, intercept = f['coef'].copy(), f['intercept'].copy()

def make_call(name, batch):
    if name == 'training_mean':
        return lambda a: np.broadcast_to(mean, (len(a), 3)).copy()
    if name == 'ridge':
        return lambda a: a.reshape(len(a), -1) @ coef.T + intercept
    model = models[name]
    shape = (batch, 128, 32) if name == 'cnn_bilstm' else (batch, 128, 32, 1)
    @tf.function(input_signature=[tf.TensorSpec(shape, tf.float32)], autograph=False)
    def forward(a):
        with tf.device(device):
            return model(a, training=False)
    def call(a):
        if name != 'cnn_bilstm':
            a = a[..., None]
        # Host tensor creation and GPU transfer are inside the timed call.
        with tf.device('/CPU:0'):
            host = tf.convert_to_tensor(a, dtype=tf.float32)
        return forward(host).numpy()  # synchronizes accelerator completion
    return call

result = {'status':'running', 'device':args.device, 'utc':datetime.now(timezone.utc).isoformat(),
          'source':str(source), 'source_sha256':hashes, 'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
          'input_shape':list(x.shape), 'target_indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),
          'hardware':{'platform':platform.platform(), 'cpu':subprocess.check_output(['lscpu'],text=True),
                      'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()},
          'versions':{'python':sys.version, 'tensorflow':tf.__version__, 'keras':tf.keras.__version__, 'numpy':np.__version__, 'scipy':scipy.__version__},
          'protocol':{'warmup_calls_per_model_batch':20, 'rounds':3, 'calls_per_round':100,
                      'batch_sizes':[1,64], 'cpu_intra_threads':4, 'cpu_inter_threads':1,
                      'boundary':'already-scaled NumPy EEG windows to completed NumPy XYZ predictions',
                      'excluded':['acquisition','filtering','scaling','window construction','model loading','graph tracing','inverse target transform'],
                      'neural_api':'tf.function(model(x, training=False)), host-to-device transfer and .numpy() synchronization included',
                      'baseline_api':'saved NumPy mean and ridge, CPU only',
                      'order':'seeded randomized model order within each round, sequential calls'},
          'models':{}, 'verification':{}}
names = list(NAMES if args.device == 'cpu' else NAMES[2:])
for batch in (1,64):
    calls = {n: make_call(n,batch) for n in names}
    for n, call in calls.items():
        for _ in range(20):
            out = call(x[:batch])
        assert out.shape == (batch,3) and np.isfinite(out).all()
    samples = {n:[] for n in names}
    rng = np.random.default_rng(42)
    round_stats = {n:[] for n in names}
    for round_no in range(3):
        for n in rng.permutation(names):
            latencies = []
            for i in range(100):
                start = ((round_no*100+i)*37) % (len(x)-batch+1)
                inp = x[start:start+batch]
                begin = time.perf_counter_ns()
                out = calls[n](inp)
                elapsed = (time.perf_counter_ns()-begin)/1e6
                latencies.append(elapsed)
            samples[n].extend(latencies)
            round_stats[n].append(float(np.median(latencies)))
            print(args.device, batch, round_no, n, 'median ms', round_stats[n][-1], flush=True)
    for n in names:
        a = np.asarray(samples[n]); assert len(a)==300 and np.isfinite(a).all() and np.all(a>0)
        entry = {'n_calls':300,'median_ms':float(np.median(a)), 'p95_ms':float(np.percentile(a,95)),
                 'mean_ms':float(a.mean()),'min_ms':float(a.min()),'round_median_ms':round_stats[n],
                 'median_ms_per_window':float(np.median(a)/batch)}
        result['models'].setdefault(n,{})[str(batch)]=entry
        np.save(args.output/f'{args.device}_{n}_batch{batch}_latency_ms.npy',a)
    if batch == 64:
        for n in names:
            # Validate a complete fixed-size first batch against the saved run.
            prediction = calls[n](x[:64])
            with np.load(source/f'{n}_predictions.npz') as saved:
                expected=saved['y_pred'][:64]
            delta=np.abs(prediction-expected)
            result['verification'][n]={'first_64_max_absolute_difference':float(delta.max()),'first_64_mean_absolute_difference':float(delta.mean())}
            np.testing.assert_allclose(prediction,expected,atol=.01,rtol=.001)
            assert delta.mean()<.001
for n, model in models.items():
    for before, after in zip(initial[n], model.get_weights()):
        np.testing.assert_array_equal(before,after)
assert hashes == {p.name:digest(p) for p in artifacts}
result['verification']['source_files_and_neural_weights_unchanged']=True
result['status']='complete'
(args.output/f'{args.device}_results.json').write_text(json.dumps(result,indent=2)+'\n')
print('COMPLETE',args.device,args.output,flush=True)
