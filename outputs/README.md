# Published pipeline results

## P1 inference latency

[CPU/GPU benchmark](inference-benchmark-p1-20261014/RESULTS.md)
· [Runner](../experiments/benchmark_inference.py)

Warmed single-window calls favor CPU for the three CNN variants, while GPU
modestly improves CNN–BiLSTM latency. All neural models have faster batch-64
throughput on GPU. Timings include host/device transfer and output synchronization
but exclude acquisition, preprocessing, loading and batch accumulation. Raw
latency arrays retain 300 calls per model/device/batch configuration.

## Category 1: C1 audit and C2 classification

[Condition audit](condition-audit-20261014T115618Z/RESULTS.md)
· [Classifier results and plots](weight-classification-p1-gpu-20261014T115817Z/RESULTS.md)
· [Methods](../documentation/CONDITION_CLASSIFICATION.md)

C1 audits 3,528 supplied trials across P1–P12, retaining series types and current/
previous trial conditions. V2's held-out current weight/surface combinations also
appear in training; series-type composition differs across splits.

C2 classifies 165/330/660 g from P1 EEG at fixed sandpaper. Whole weight runs 1/4
train, 7 validates, and 8/9 test. Each full trial produces one prediction from
its mean window probabilities. Training-only scaling and balanced trial/class
training weights apply. CNN + ELU runs on GPU; logistic and majority use CPU.

| Classifier | Test trials | Balanced accuracy | Macro F1 | Circular upper-tail fraction |
| --- | ---: | ---: | ---: | ---: |
| Training majority | 68 | 33.33% | 0.1685 | — |
| Logistic | 68 | 32.48% | 0.3237 | 0.592 |
| CNN + ELU | 68 | 27.73% | 0.2329 | 0.924 |

No reliable weight discrimination appears in this first setup. Circular-shift
references retain predictions and shift whole-trial labels within test runs;
they do not retrain under shuffled labels. Both two-classifier adjusted fractions
are 1.0. The report preserves learning history, confusion matrices, all per-run
scores and [verification](weight-classification-p1-gpu-20261014T115817Z/verification.json).

## Saved-decoder analyses: P1 event-aligned diagnostics

[Results and plots](saved-decoder-analysis-p1-20261013T172654Z/RESULTS.md) · [Methods](../documentation/SAVED_DECODER_ANALYSES.md)

The initial P1 analysis of **Project Idea Category 2: Additional Results from
Existing Decoders** reconstructs saved predictions for six P1 reference
models without training or new inference.
They align all 68 held-out trials to raw events, plot coordinates and signed
error over time, and summarize phase, motion, density and bias. Exact sample
matches verify a two-second trial-metadata/continuous-recording clock offset.
CNN–BiLSTM errors are higher during reaching and return than preparation or the
airborne interval. Pure-window counts are sparse for short phases, and reduced
coordinates do not provide physical distance errors.

## Frozen P1 cross-participant evaluation

[Run report](frozen-p1-cross-participant-gpu-20260930T125956Z/RESULTS.md) · [Methods](../documentation/FROZEN_P1_TRANSFER.md)

All six saved P1 models and P1 EEG/target scalers are frozen. P2–P12 receive no
fitting or adaptation. Primary evaluation uses weight runs 8–9 (45,389 windows;
fixed sandpaper throughout); all nine target runs are a separately labeled secondary
evaluation. Neural inference uses GPU, with mean/ridge on CPU.

| Model | Participant-macro R² X | R² Y | R² Z |
| --- | ---: | ---: | ---: |
| Training mean | -0.303 | -0.535 | -0.244 |
| Ridge | -0.825 | -0.943 | -0.461 |
| CNN | -0.722 | -0.884 | -0.418 |
| CNN–BiLSTM | -1.060 | -1.163 | -0.585 |
| CNN + ELU | -0.641 | -0.687 | -0.322 |
| EEGNet regression | -0.578 | -0.698 | -0.324 |

The P1 replay control preserves its positive learned-model scores, while frozen
transfer is poor. Only P10 Z for ELU/EEGNet is slightly positive. Every model's
macro average is negative; the P1 mean performs better than the learned models
on these averages. All raw participant recordings confirm the wrist mapping.
This is single-source transfer on existing reduced files, whose upstream
processing and physical units remain unresolved. Saved source artifacts and
neural weights are unchanged.

## Six-model P1 comparison

[Run report](six-model-comparison-p1-gpu-20260930T095506Z/RESULTS.md) · [Methods](../documentation/SIX_MODEL_COMPARISON.md)

This comparison uses the fixed v2 P1 run split and preprocessing. Four neural
models trained on GPU; mean and ridge used CPU. Ridge regularization was selected
using validation run 7; all models use test runs 8–9 only for final evaluation.

| Model | Wrist X R² | Wrist Y R² | Wrist Z R² |
| --- | ---: | ---: | ---: |
| Training mean | -0.204 | -0.324 | -0.185 |
| Ridge | 0.558 | 0.420 | 0.462 |
| Original CNN | 0.547 | 0.399 | 0.451 |
| CNN–BiLSTM | 0.809 | 0.783 | 0.748 |
| CNN + ELU | 0.746 | 0.713 | 0.678 |
| EEGNet regression | 0.758 | 0.683 | 0.663 |

The ELU-only ablation improves all axes relative to the affine CNN in this run.
EEGNet regression improves over the original CNN but does not consistently exceed
the ELU-only variant. This is one participant and seed, with friction condition
3 only in test. It does not establish statistical significance or population
performance. CNN–BiLSTM was retrained with a per-model seed reset, so its score
is distinct from the historical v2 run below.

## Pipeline v1: Original baseline

These are reproductions of the original notebook before preprocessing, model, or split corrections. The saved CPU and GPU runs are labeled by execution device; neither is a cross-subject evaluation.

| Run | CNN training | CNN–BiLSTM training | CNN–BiLSTM R² (targets 1 / 2 / 3) |
|---|---:|---:|---|
| [Windows CPU](original-p1-cpu/RESULTS.md) | 259.86 s | 272.34 s | 0.833 / 0.823 / 0.770 |
| [WSL2 RTX 3060 GPU](original-p1-gpu/RESULTS.md) | 36.31 s | 153.57 s | 0.834 / 0.839 / 0.729 |

The GPU run completed training in 189.89 seconds versus 532.20 seconds on CPU (about 2.8× faster in these single runs). Python/platform environments differed and model initialization was unseeded, so this is an observed timing comparison rather than a controlled benchmark.

The [baseline methods and limitations](../documentation/ORIGINAL_BASELINE.md) explain window construction, architectures, the split, and why these test scores cannot yet support claims about unseen subjects. Each run page links its learning curves, target scores, and prediction scatter plot.

The public `results.json` files preserve complete numeric histories. Raw predictions, model weights, logs, and notebook execution snapshots remain local.

## Pipeline v2: Corrected Baseline

The separate [v2 notebook](../pipeline_v2.ipynb) fixes the first evaluation issues while retaining the two original model architectures. Its `PARTICIPANT_ID` setting selects an aligned participant; the central [changelog](../CHANGELOG.md) summarizes the version differences, and the [v2 methods](../documentation/PIPELINE_V2_CORRECTED_BASELINE.md) explain each correction and its remaining limits. For the supplied nine-run participants, it trains on runs 1–6, validates on run 7, and tests on runs 8–9. EEG channels and targets are standardized from training samples only; windows never cross run boundaries.

The [first GPU result for P1](pipeline-v2-corrected-baseline-p1-gpu-20260929T173627Z/RESULTS.md) reports CNN–BiLSTM R² of **0.784 / 0.757 / 0.726** and CNN R² of **0.547 / 0.399 / 0.451** for wrist X/Y/Z. Its run page includes learning curves, prediction plots, and scores for each held-out run separately. Both P1 test runs contain friction condition 3 only. These scores and the original random-window scores measure different evaluation settings.
