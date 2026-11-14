"""Run and score the C_5 model comparison on verified event-aligned windows."""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from threadpoolctl import threadpool_limits

ROOT = Path.cwd()
ORIGINAL = ROOT / "outputs/c5-stage-specific-ridge-20261114"
sys.path[:0] = [str(ORIGINAL), str(ROOT / "experiments/07_timing_alignment")]
import ridge_reference as ridge
from run_decoder_gap_preferences import build_model
from run_timing_alignment import build_cnn

MODELS = ("cnn_elu", "eegnet_regression", "cnn_bilstm")
PEOPLE = tuple(f"P{i}" for i in range(1, 13))
SEED = 42
EPOCHS = 15
BATCH = 64


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def evaluation(y, trial, series, stage, pure, policy):
    groups = {}
    eligible = (series >= 8) & (pure if policy == "Pure stage" else True)
    for code, name in enumerate(ridge.STAGES, 1):
        indices = np.flatnonzero(eligible & (stage == code))
        ids, inverse, counts = np.unique(trial[indices], return_inverse=True, return_counts=True)
        if len(ids) < ridge.MIN_TEST_TRIALS:
            continue
        w = 1.0 / (len(ids) * counts[inverse])
        truth = y[indices].astype(np.float64)
        center = np.average(truth, axis=0, weights=w)
        variance = np.average((truth - center)**2, axis=0, weights=w)
        groups[name] = (indices, truth, w, variance, len(ids))
    return groups


def scored(person, policy, architecture, scope, train_stage, name, group, mse):
    indices, _, _, variance, trials = group
    r2 = np.divide(mse, variance, out=np.full(3, np.nan), where=variance > 1e-15)
    r2 = 1.0 - r2
    row = dict(participant=person, policy=policy, architecture=architecture, scope=scope,
               train_stage=train_stage, test_stage=name, windows=len(indices), trials=trials,
               mse=float(np.mean(mse)), r2=float(np.mean(r2)) if np.isfinite(r2).all() else "")
    for i, axis in enumerate("xyz"):
        row[f"mse_{axis}"] = float(mse[i])
        row[f"r2_{axis}"] = float(r2[i]) if np.isfinite(r2[i]) else ""
        row[f"variance_{axis}"] = float(variance[i])
    return row


class GpuFit:
    def __init__(self, architecture):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        self.model = build_cnn() if architecture == "cnn_elu" else build_model(tf, architecture)
        self.input_shape = (128, 32) if architecture == "cnn_bilstm" else (128, 32, 1)
        self.optimizer = self.model.optimizer
        self.optimizer.build(self.model.trainable_variables)
        self.initial = self.model.get_weights()
        self.optimizer_initial = [variable.numpy().copy() for variable in self.optimizer.variables]
        self.random_states = [(g.state, g.state.numpy().copy()) for layer in self.model._flatten_layers()
                              for g in getattr(layer, "_seed_generators", ())]

    @tf.function(reduce_retracing=True, jit_compile=False)
    def step(self, x, y):
        with tf.GradientTape() as tape:
            error = self.model(x, training=True) - y
            loss = tf.reduce_mean(error**2)
        self.optimizer.apply_gradients(zip(tape.gradient(loss,self.model.trainable_variables),
                                          self.model.trainable_variables))
        return loss

    @tf.function(reduce_retracing=True, jit_compile=False)
    def inference(self, x):
        return self.model(x, training=False)

    def predict(self, x, y, train, test):
        self.model.set_weights(self.initial)
        for var, value in zip(self.optimizer.variables, self.optimizer_initial):
            var.assign(value)
        for state, value in self.random_states:
            state.assign(value)
        xx = x.reshape((-1, *self.input_shape))
        rng = np.random.default_rng(SEED)
        for _ in range(EPOCHS):
            order = rng.permutation(train)
            for i in range(0,len(order),BATCH):
                ids = order[i:i+BATCH]
                loss = self.step(xx[ids],y[ids])
        assert np.isfinite(loss.numpy())
        pred = np.concatenate([self.inference(xx[test[i:i+256]]).numpy() for i in range(0,len(test),256)])
        assert pred.shape == (len(test),3) and np.isfinite(pred).all()
        return pred, self.model.count_params()


def one(person, meta, out, original):
    x, y, series, stage, trial, pure, condition, _speed, _checks = ridge.extract(person, meta)
    x = x.reshape(-1,128,32)
    groups = {policy:evaluation(y,trial,series,stage,pure,policy) for policy in ridge.POLICIES}
    scores, fits = [], []
    saved = defaultdict(list)
    for row in original:
        saved[row["policy"],row["model"],row["train_stage"],row["test_stage"]].append(row)
    for (policy,kind,tr,te), rows in saved.items():
        group = groups[policy][te]
        assert all(int(row["windows"])==len(group[0]) and int(row["trials"])==group[4] for row in rows)
        mse = np.array([[float(row[f"mse_{a}"]) for a in "xyz"] for row in rows]).mean(axis=0)
        if kind=="Matched pooled ridge":
            continue
        assert len(rows)==1
        scope={"Pooled ridge":"Pooled","Matched pooled ridge":"Matched pooled",
               "Specialist ridge":"Specialist","Stage mean":"Stage mean","Pooled mean":"Pooled mean"}[kind]
        scores.append(scored(person,policy,"ridge" if "ridge" in kind else "mean",scope,tr,te,group,mse))

    alpha = float(meta["gaps"]["1"]["ridge"]["selected_alpha"])
    strata = series * 100 + condition
    for rule, evaluation_groups in groups.items():
        eligible = pure if rule == "Pure stage" else np.ones_like(pure)
        train = np.flatnonzero(eligible & (series <= 6))
        for code, stage_name in enumerate(ridge.STAGES, 1):
            subset = train[stage[train] == code]
            if (rule, "Specialist ridge", stage_name, stage_name) not in saved:
                continue
            group = evaluation_groups[stage_name]
            errors = []
            flat = x.reshape(len(x), -1)
            for seed in ridge.SEEDS:
                matched = ridge.class_matched_indices(train, subset, strata, seed)
                model = ridge.Ridge(alpha=alpha, solver="lsqr", tol=1e-3, max_iter=300)
                model.fit(flat[matched], y[matched])
                assert max(model.n_iter_) < 300
                residual = model.predict(flat[group[0]]).astype(np.float64) - group[1]
                errors.append(np.average(residual**2, axis=0, weights=group[2]))
            scores.append(scored(person, rule, "ridge", "Matched pooled", stage_name, stage_name,
                                 group, np.mean(errors, axis=0)))

    policy = "Pure stage"
    train = np.flatnonzero(pure & (series <= 6))
    test = np.flatnonzero(pure & (series >= 8))
    strata=series*100+condition
    test_rows = {name:np.searchsorted(test, group[0]) for name,group in groups[policy].items()}
    assert all(np.array_equal(test[test_rows[name]],group[0]) for name,group in groups[policy].items())
    specialists = {}
    for code,name in enumerate(ridge.STAGES,1):
        stage_train = train[stage[train]==code]
        if len(stage_train)>=ridge.MIN_TRAIN_WINDOWS and len(np.unique(trial[stage_train]))>=ridge.MIN_TRAIN_TRIALS:
            assert any(k[:3]==(policy,"Specialist ridge",name) for k in saved)
            specialists[name]=stage_train

    def train_score(architecture, scope, stage_name, subset, names):
        with tf.device("/GPU:0"):
            pred, params = trainer.predict(x, y, subset, test)
        fits.append(dict(participant=person, architecture=architecture, scope=scope, train_stage=stage_name,
                         n_train=len(subset), n_train_trials=len(np.unique(trial[subset])), parameters=params,
                         seed=SEED, epochs=EPOCHS, batch_size=BATCH, device="GPU:0"))
        for name in names:
            group=groups[policy][name]
            error=pred[test_rows[name]].astype(np.float64)-group[1]
            mse=np.average(error**2,axis=0,weights=group[2])
            scores.append(scored(person,policy,architecture,scope,stage_name,name,group,mse))

    for architecture in MODELS:
        with tf.device("/GPU:0"):
            trainer = GpuFit(architecture)
        train_score(architecture,"Pooled","",train,groups[policy])
        for stage_name, subset in specialists.items():
            train_score(architecture,"Specialist",stage_name,subset,groups[policy])
            if stage_name in groups[policy]:
                matched=ridge.class_matched_indices(train,subset,strata,SEED)
                train_score(architecture,"Matched pooled",stage_name,matched,(stage_name,))
        print("ARCH_COMPLETE",person,architecture,len(fits),flush=True)
    ridge.write_csv(out/f"{person}-fits.csv",fits)
    ridge.write_csv(out/f"{person}-scores.csv",scores)
    print("PARTICIPANT_COMPLETE",person,flush=True)


def main(out, person):
    assert person in PEOPLE
    if (out/f"{person}-scores.csv").is_file():
        return
    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("C_5 CNN fitting requires TensorFlow CUDA.")
    tf.config.experimental.set_memory_growth(gpu[0],True)
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    with tf.device("/GPU:0"):
        probe = tf.linalg.matmul(tf.ones((32,32)),tf.ones((32,32)))
        assert probe.device.endswith("GPU:0")
    original = [r for r in read_csv(ORIGINAL/"metrics.csv") if r["participant"]==person]
    assert original
    meta = json.loads((ridge.REFERENCE/"results.json").read_text(encoding="utf-8"))["participant_results"][person]
    with threadpool_limits(3):
        one(person,meta,out,original)

if __name__=="__main__":
    main(Path(sys.argv[1]),sys.argv[2])
