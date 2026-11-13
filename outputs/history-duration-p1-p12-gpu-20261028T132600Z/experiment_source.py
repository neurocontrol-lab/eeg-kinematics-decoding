"""EXP-018: fixed-bin histories, separately trained P1–P12 ridge/CNN models."""
import os
for key, value in {'MPLBACKEND': 'Agg', 'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
                   'TF_CPP_MIN_LOG_LEVEL': '2', 'OPENBLAS_NUM_THREADS': '4',
                   'OMP_NUM_THREADS': '4'}.items():
    os.environ.setdefault(key, value)
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/eeg_kinematics/paths.py').is_file())
sys.path.insert(0, str(ROOT/'src'))
from eeg_kinematics.paths import REDUCED_DATA_DIR
from eeg_kinematics.history_duration import HISTORIES, matched_targets, mean_bin_windows

PROTOCOL = ROOT/'documentation/methods/HISTORY_DURATION.md'
HELPER = ROOT/'src/eeg_kinematics/history_duration.py'
T1_SOURCE = ROOT/'experiments/07_timing_alignment/run_timing_alignment.py'
spec = importlib.util.spec_from_file_location('t1_reference', T1_SOURCE)
t1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t1)
build_cnn, metrics = t1.build_cnn, t1.metrics
SPLITS = {'train': list(range(1, 7)), 'validation': [7], 'test': [8, 9]}
ALPHAS = (1., 10., 100., 1000., 10000.)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_participant(pid):
    eeg = np.asarray(loadmat(REDUCED_DATA_DIR/f'eeg_P{pid}.mat')['eeg'].T, dtype=np.float32, order='C')
    kin = np.asarray(loadmat(REDUCED_DATA_DIR/f'kin_P{pid}.mat')['kin'], dtype=np.float32, order='C')
    starts = loadmat(REDUCED_DATA_DIR/f'session_start_P{pid}.mat')['session_start'].ravel().astype(int)-1
    ends = np.r_[starts[1:], len(eeg)]
    assert eeg.shape == (len(kin), 32) and kin.shape[1] == 3 and len(starts) == 9
    assert starts[0] == 0 and np.all(np.diff(starts) > 0)
    assert np.isfinite(eeg).all() and np.isfinite(kin).all()
    bounds = {r: (int(lo), int(hi)) for r, (lo, hi) in enumerate(zip(starts, ends), 1)}
    es, ks = StandardScaler(), StandardScaler()
    for r in SPLITS['train']:
        lo, hi = bounds[r]
        es.partial_fit(eeg[lo:hi]); ks.partial_fit(kin[lo:hi])
    scaling = {'eeg_mean': es.mean_.tolist(), 'eeg_scale': es.scale_.tolist(),
               'target_mean': ks.mean_.tolist(), 'target_scale': ks.scale_.tolist(),
               'samples_seen': int(es.n_samples_seen_), 'fit_series': SPLITS['train']}
    return es.transform(eeg, copy=False), ks.transform(kin, copy=False), bounds, scaling


def population_summary(participants):
    rows = []
    for pid, p in participants.items():
        for model in ('ridge', 'cnn_elu'):
            baseline = p['test'][f'{model}_128']['aggregate']['mse_mean']
            for history in (256, 512):
                longer = p['test'][f'{model}_{history}']['aggregate']['mse_mean']
                rows.append({'participant': pid, 'model': model, 'history_ms': history*2,
                             'baseline_mse': baseline, 'longer_mse': longer,
                             'longer_minus_baseline': longer-baseline,
                             'relative_mse_reduction_pct': 100*(baseline-longer)/baseline})
    rng = np.random.default_rng(20261028)
    draws = rng.integers(0, 12, size=(20000, 12))
    summary = {}
    for model in ('ridge', 'cnn_elu'):
        for ms in (512, 1024):
            selected = [r for r in rows if r['model'] == model and r['history_ms'] == ms]
            rel = np.array([r['relative_mse_reduction_pct'] for r in selected])
            delta = np.array([r['longer_minus_baseline'] for r in selected])
            assert len(rel) == 12
            summary[f'{model}_{ms}ms'] = {
                'participants': 12, 'improved': int((rel > 0).sum()),
                'mean_relative_mse_reduction_pct': float(rel.mean()),
                'median_relative_mse_reduction_pct': float(np.median(rel)),
                'mean_relative_reduction_bootstrap_95': np.quantile(rel[draws].mean(axis=1), [.025, .975]).tolist(),
                'mean_mse_difference': float(delta.mean()),
                'mean_mse_difference_bootstrap_95': np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()}
    return rows, summary


def render(out, result):
    rows, summary = population_summary(result['participants'])
    result['population'] = summary
    with (out/'participant_effects.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, ax in zip(('ridge', 'cnn_elu'), axes):
        for ms, marker in [(512, 'o'), (1024, 's')]:
            values = [r['relative_mse_reduction_pct'] for r in rows if r['model'] == model and r['history_ms'] == ms]
            ax.plot(range(1, 13), values, marker=marker, label=f'{ms} ms vs 256 ms')
        ax.axhline(0, color='black', linewidth=.8); ax.set_xticks(range(1, 13))
        ax.set(title=model, xlabel='Participant', ylabel='Test MSE reduction (%) — positive is better')
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'participant_effects.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for model, ax in zip(('ridge', 'cnn_elu'), axes):
        for pid, p in result['participants'].items():
            ax.plot([256, 512, 1024], [p['histories'][str(h)][model]['validation']['mse_mean'] for h in HISTORIES], alpha=.5, marker='.', label=pid)
        ax.set(title=f'{model}: validation, all participants', xlabel='History (ms)', ylabel='MSE (own training-standardized wrist units)')
        ax.set_xticks([256, 512, 1024]); ax.grid(alpha=.2)
    axes[1].legend(ncol=3, fontsize=8); fig.tight_layout(); fig.savefig(out/'validation_histories.png', dpi=160); plt.close(fig)
    lines = ['# EXP-018: EEG history duration across P1–P12', '',
             'Separate within-participant ridge/CNN + ELU models; fixed 128 × 32 mean-bin inputs, 2-ms end gap and shared target indices. Longer history also increases averaging.', '',
             '## Paired participant effects', '',
             'Positive relative MSE reduction favours longer history. Participants have equal weight; intervals are exploratory 95% participant bootstrap intervals, not independent-window tests.', '',
             '| Model | History vs 256 ms | Improved | Mean MSE reduction (%) | Median (%) | Mean reduction interval |',
             '|---|---:|---:|---:|---:|---|']
    for key, item in summary.items():
        lo, hi = item['mean_relative_reduction_bootstrap_95']
        lines.append(f"| {key.rsplit('_',1)[0]} | {key.rsplit('_',1)[1]} | {item['improved']}/12 | {item['mean_relative_mse_reduction_pct']:.2f} | {item['median_relative_mse_reduction_pct']:.2f} | [{lo:.2f}, {hi:.2f}] |")
    lines += ['', '![Paired effects](participant_effects.png)', '', '![Validation histories](validation_histories.png)', '',
              '## Individual test scores', '', '| Participant | Model | History (ms) | MSE | R² X | R² Y | R² Z |', '|---|---|---:|---:|---:|---:|---:|']
    for pid, p in result['participants'].items():
        for model in ('ridge', 'cnn_elu'):
            for h in HISTORIES:
                m = p['test'][f'{model}_{h}']['aggregate']
                lines.append(f"| {pid} | {model} | {h*2} | {m['mse_mean']:.6f} | "+' | '.join(f'{v:.4f}' if v is not None else 'n.a.' for v in m['r2'])+' |')
    lines += ['', '## Limits', '',
              'This is a group summary of separately trained subject-specific decoders, not a pooled population model or unseen-participant generalization. All candidates and test comparisons were declared in advance. Ridge alpha and secondary history choices use validation only. One seed per configuration; familiar participants/test series; unknown upstream EEG preprocessing.', '',
              'Increasing history changes smoothing, temporal resolution, effective sample rate and the physical support of CNN filters. This experiment cannot identify a pure context-length effect without full-resolution/averaging-only controls. MSE uses each participant’s own training-standardized targets, not physical distances. The four bootstrap contrasts are exploratory and are not multiplicity-adjusted.', '',
              '[Declared protocol](declared_protocol.md) · [Frozen selections](selection.json) · [Public registry](https://docs.google.com/spreadsheets/d/1Q2nJqr66ftuz7CP9dMiYRdE2RGwavmGbMGDkOS7yiWg/edit)']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path); ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    out = (args.output_dir or ROOT/'outputs'/('history-duration-p1-p12-gpu-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    assert out.parent == (ROOT/'outputs').resolve()
    source_paths = {'runner': Path(__file__), 'helper': HELPER, 'protocol': PROTOCOL, 'reference_model': T1_SOURCE}
    sources = {k: sha(p) for k, p in source_paths.items()}
    if args.resume:
        result = json.loads((out/'results.json').read_text())
        assert result['source_sha256'] == sources and result['status'] != 'complete'
    else:
        out.mkdir(exist_ok=False)
        result = {'experiment_id': 'EXP-018', 'status': 'running', 'phase': 'fitting', 'created_at': stamp(),
                  'histories_samples': list(HISTORIES), 'histories_ms': [h*2 for h in HISTORIES],
                  'input_shape': [128, 32], 'end_gap_samples': 1, 'stride_samples': 64, 'seed': 42,
                  'runs': SPLITS, 'source_sha256': sources, 'participants': {}}
        for key, name in [('runner', 'experiment_source.py'), ('helper', 'history_source.py'), ('protocol', 'declared_protocol.md'), ('reference_model', 'reference_model_source.py')]:
            (out/name).write_bytes(source_paths[key].read_bytes())
    def save(): save_json(out/'results.json', result)
    print('OUTPUT_DIRECTORY', out, flush=True); save()
    gpu = tf.config.list_physical_devices('GPU'); assert gpu, 'GPU required'
    for device in gpu: tf.config.experimental.set_memory_growth(device, True)
    tf.config.threading.set_intra_op_parallelism_threads(4); tf.config.threading.set_inter_op_parallelism_threads(2)
    with tf.device('/GPU:0'): probe = tf.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
    assert 'GPU:0' in probe.device; probe.numpy()
    result['gpu'] = tf.config.experimental.get_device_details(gpu[0]); result['gpu_probe_device'] = probe.device
    result['versions'] = {'python': sys.version.split()[0], 'tensorflow': tf.__version__, 'numpy': np.__version__}
    for pid in range(1, 13):
        label = f'P{pid}'
        if result['participants'].get(label, {}).get('fitting_complete'): continue
        print('FIT_PARTICIPANT', label, flush=True)
        eeg, kin, bounds, scaling = load_participant(pid)
        p = result['participants'].setdefault(label, {'histories': {}, 'test': {}})
        p['scaling'] = scaling; p['bounds'] = {str(r): list(b) for r, b in bounds.items()}
        p['input_sha256'] = {n: sha(REDUCED_DATA_DIR/n) for n in (f'eeg_P{pid}.mat', f'kin_P{pid}.mat', f'session_start_P{pid}.mat')}
        splits = {name: matched_targets(bounds, runs) for name, runs in SPLITS.items()}
        p['split_counts'] = {name: len(t) for name, (t, _) in splits.items()}
        np.savez_compressed(out/f'{label}_indices.npz', **{f'{name}_{key}': a for name, pair in splits.items() for key, a in zip(('targets', 'run_ids'), pair)})
        ys = {name: kin[t] for name, (t, _) in splits.items()}
        for h in HISTORIES:
            entry = p['histories'].setdefault(str(h), {})
            if 'ridge' in entry and 'cnn_elu' in entry: continue
            train = mean_bin_windows(eeg, *splits['train'], bounds, h)
            val = mean_bin_windows(eeg, *splits['validation'], bounds, h)
            if 'ridge' not in entry:
                candidates = []
                for alpha in ALPHAS:
                    ridge = Ridge(alpha=alpha, solver='lsqr', tol=.001, max_iter=200).fit(train.reshape(len(train), -1), ys['train'])
                    pred = ridge.predict(val.reshape(len(val), -1)); name = f'{label}_ridge_h{h}_a{alpha:g}.npz'
                    np.savez_compressed(out/name, coef=ridge.coef_, intercept=ridge.intercept_, validation_pred=pred)
                    candidates.append({'alpha': alpha, 'validation': metrics(ys['validation'], pred), 'weights': name, 'n_iter': ridge.n_iter_.tolist()})
                best = min(candidates, key=lambda c: (c['validation']['mse_mean'], c['alpha']))
                entry['ridge'] = {**best, 'candidates': candidates}; save()
                print('RIDGE_DONE', label, h*2, 'ms', 'val_mse', best['validation']['mse_mean'], flush=True)
            if 'cnn_elu' not in entry:
                tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(42)
                with tf.device('/GPU:0'):
                    cnn = build_cnn()
                    initial = hashlib.sha256(b''.join(w.tobytes() for w in cnn.get_weights())).hexdigest()
                    assert initial == result.setdefault('cnn_initial_sha256', initial)
                    hist = cnn.fit(train[..., None], ys['train'], validation_data=(val[..., None], ys['validation']), epochs=15, batch_size=64, verbose=0)
                    pred = cnn.predict(val[..., None], batch_size=64, verbose=0)
                    name = f'{label}_cnn_elu_h{h}.keras'; cnn.save(out/name)
                np.savez_compressed(out/f'{label}_cnn_elu_h{h}_validation.npz', y_pred=pred)
                entry['cnn_elu'] = {'validation': metrics(ys['validation'], pred), 'weights': name, 'initial_sha256': initial,
                                    'epochs': 15, 'parameters': cnn.count_params(), 'history': {k: [float(v) for v in a] for k, a in hist.history.items()}}
                save(); print('CNN_DONE', label, h*2, 'ms', 'val_mse', entry['cnn_elu']['validation']['mse_mean'], flush=True)
                del cnn
            del train, val; gc.collect()
        p['selection'] = {m: min(HISTORIES, key=lambda h: (p['histories'][str(h)][m]['validation']['mse_mean'], h)) for m in ('ridge', 'cnn_elu')}
        p['fitting_complete'] = True; save()
        del eeg, kin, ys; gc.collect()
    if not (out/'selection.json').exists():
        frozen = {'frozen_at': stamp(), 'models': {pid: p['selection'] for pid, p in result['participants'].items()},
                  'scope': 'All 12 fitting phases complete before any test prediction; every history test is predeclared.'}
        save_json(out/'selection.json', frozen)
    frozen = json.loads((out/'selection.json').read_text())
    assert frozen['models'] == {pid: p['selection'] for pid, p in result['participants'].items()}
    result['selection_frozen_at'] = frozen['frozen_at']; result['phase'] = 'testing'; save()
    print('ALL_SELECTIONS_FROZEN', frozen['models'], flush=True)
    for pid in range(1, 13):
        label = f'P{pid}'; p = result['participants'][label]
        eeg, kin, bounds, scaling = load_participant(pid); assert scaling == p['scaling']
        targets, ids = matched_targets(bounds, SPLITS['test']); y = kin[targets]
        for h in HISTORIES:
            x = mean_bin_windows(eeg, targets, ids, bounds, h)
            for model in ('ridge', 'cnn_elu'):
                key = f'{model}_{h}'
                if key in p['test']: continue
                e = p['histories'][str(h)][model]
                if model == 'ridge':
                    with np.load(out/e['weights']) as z: pred = x.reshape(len(x), -1) @ z['coef'].T + z['intercept']
                else:
                    tf.keras.backend.clear_session()
                    with tf.device('/GPU:0'):
                        cnn = tf.keras.models.load_model(out/e['weights'], compile=False)
                        pred = cnn.predict(x[..., None], batch_size=64, verbose=0)
                    del cnn
                name = f'{label}_{key}_test.npz'
                np.savez_compressed(out/name, y_true=y, y_pred=pred, target_sample_index=targets, run_id=ids)
                p['test'][key] = {'predictions': name, 'aggregate': metrics(y, pred),
                                   'by_series': {str(r): metrics(y[ids == r], pred[ids == r]) for r in (8, 9)}}
                save()
            del x
        print('TEST_DONE', label, flush=True)
        del eeg, kin; gc.collect()
    assert sources == {k: sha(p) for k, p in source_paths.items()}
    result['source_files_unchanged'] = True
    render(out, result)
    result['status'] = 'complete'; result['phase'] = 'complete'; result['completed_at'] = stamp(); save()
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
