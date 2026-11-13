# Published pipeline results

## T1 — Timing preferences and held-out comparisons across decoders

[Validation gap preferences](timing-decoder-preferences-p1-p12-20261111T113206Z/RESULTS.md) · [Held-out summary](timing-decoder-preferences-p1-p12-20261111T113206Z/heldout/population_summary.json) · [Held-out verification](timing-decoder-preferences-p1-p12-20261111T113206Z/heldout/verification.json)

The validation analysis compares four decoder families across six end gaps. A separate held-out analysis compares validation-selected gaps with the matched 2-ms reference. For the added decoders, EEGNet regression has lower mean test MSE with the selected gap; the CNN–BiLSTM paired interval spans zero. Both added neural decoders use one seed.

## R6 — Movement dynamics and error across P1–P12

[Error models](r6-population-error-models-20261111/RESULTS.md) · [Verification](r6-population-error-models-20261111/verification.json) · [Timing curves and scatter follow-up](t1-r6-followup-20261110/RESULTS.md)

Adding velocity and acceleration to position and stage did not improve average held-out error explanation for ridge or CNN + ELU under the primary setup. The scatter follow-up describes movement composition within error regions; it does not establish a causal explanation or an improved wrist decoder.

## T7 — Wrist position and velocity across P1–P12

[Results and paired plot](position-velocity-p1-p12-20261110T171409Z/RESULTS.md) · [Protocol](../documentation/methods/POSITION_VELOCITY.md) · [Verification](position-velocity-p1-p12-20261110T171409Z/verification.json)

Separate participant models use identical EEG windows and target timestamps for position and velocity. Mean test skill relative to each target’s training mean is 0.230 / 0.080 for ridge and 0.362 / 0.089 for CNN + ELU. The report also gives per-axis R², which uses a different reference.

## Population T1: timing across P1–P12

[Report and plots](timing-alignment-population-gpu-20261028T133000Z/RESULTS.md) · [Protocol](../documentation/methods/POPULATION_TIMING_ALIGNMENT.md) · [Saved-weight verification](timing-alignment-population-gpu-20261028T133000Z/verification.json) · [Population-summary verification](timing-alignment-population-gpu-20261028T133000Z/population_verification.json)

Both ridge and CNN + ELU cover six end gaps with fixed 256-ms inputs. CNN mean test MSE falls 3.42%, with 9/12 participants improving; ridge's paired interval spans zero. CNN seeds are averaged within participant before equal-participant reporting. This is within-participant replication across the dataset, with exploratory participant-bootstrap intervals.

## T2: fixed-bin EEG histories across P1–P12

[Results](history-duration-p1-p12-gpu-20261028T132600Z/RESULTS.md) · [Group summary](history-duration-p1-p12-gpu-20261028T132600Z/group_summary.json) · [Verification](history-duration-p1-p12-gpu-20261028T132600Z/verification.json)

Separate ridge and CNN + ELU models compare 256/512/1,024-ms histories at fixed 128 × 32 input dimensions and a 2-ms end gap. Equal-weight participant effects are reported; this is not a pooled population model. Context and averaging change together.

## R6: movement dynamics and decoder error

[Report and plots](error-dynamics-p1-20261028T091450Z/RESULTS.md) · [Protocol](../documentation/methods/ERROR_DYNAMICS.md) · [Verification](error-dynamics-p1-20261028T091450Z/verification.json)

P1 retrospective analysis reuses six saved decoders. Speed describes some scatter streaks, but the lead CNN–BiLSTM error explainer gains no clear benefit from adding dynamics beyond position and stage: primary MSE worsens 1.20%, with a paired interval spanning zero. The vertical Z streak also includes slow movement. These are explanatory error scores, not improved wrist-decoding results.

## T1: matched-target timing alignment

[P1 timing report](timing-alignment-p1-gpu-20261027T120222Z/RESULTS.md) Â· [Protocol](../documentation/methods/TIMING_ALIGNMENT.md) Â· [Verification](timing-alignment-p1-gpu-20261027T120222Z/verification.json)

Ridge and CNN + ELU compared 2/50/100/200-ms gaps with fixed 256-ms input windows and identical target timestamps. Validation selected 2 ms for ridge and 100 ms for CNN. The selected CNN gap worsened held-out mean MSE by 13.15% relative to its retrained 2-ms reference, despite improving validation MSE. Stage effects differ. This is a verified exploratory negative result for adopting an earlier-ending window, not proof of an optimal physiological lag.

## Coordinate-axis verification

[20 October audit](vertical-axis-audit-20261020/REPORT.md) Â· [Coordinate definitions](../documentation/dataset/COORDINATE_SYSTEM.md)

Position Z is vertical, positive upward. All 3,528 object trajectories rise and fall in Z across lift and set-down; the force sensors use X for lifting and Z for grip. Numeric evidence, a sample trajectory, and the exact executed audit source are retained.


## P1 inference latency

[CPU/GPU benchmark](inference-benchmark-p1-20261014/RESULTS.md)
Â· [Runner](../experiments/06_inference_benchmark/benchmark_inference.py)

Warmed single-window calls favor CPU for the three CNN variants, while GPU
modestly improves CNNâ€“BiLSTM latency. All neural models have faster batch-64
throughput on GPU. Timings include host/device transfer and output synchronization
but exclude acquisition, preprocessing, loading and batch accumulation. Raw
latency arrays retain 300 calls per model/device/batch configuration.

## Category 1: C1 audit and C2 classification

[Condition audit](condition-audit-20261014T115618Z/RESULTS.md)
Â· [Classifier results and plots](weight-classification-p1-gpu-20261014T115817Z/RESULTS.md)
Â· [Methods](../documentation/methods/CONDITION_CLASSIFICATION.md)

C1 audits 3,528 supplied trials across P1â€“P12, retaining series types and current/
previous trial conditions. V2's held-out current weight/surface combinations also
appear in training; series-type composition differs across splits.

C2 classifies 165/330/660 g from P1 EEG at fixed sandpaper. Whole weight runs 1/4
train, 7 validates, and 8/9 test. Each full trial produces one prediction from
its mean window probabilities. Training-only scaling and balanced trial/class
training weights apply. CNN + ELU runs on GPU; logistic and majority use CPU.

| Classifier | Test trials | Balanced accuracy | Macro F1 | Circular upper-tail fraction |
| --- | ---: | ---: | ---: | ---: |
| Training majority | 68 | 33.33% | 0.1685 | â€” |
| Logistic | 68 | 32.48% | 0.3237 | 0.592 |
| CNN + ELU | 68 | 27.73% | 0.2329 | 0.924 |

No reliable weight discrimination appears in this first setup. Circular-shift
references retain predictions and shift whole-trial labels within test runs;
they do not retrain under shuffled labels. Both two-classifier adjusted fractions
are 1.0. The report preserves learning history, confusion matrices, all per-run
scores and [verification](weight-classification-p1-gpu-20261014T115817Z/verification.json).

## Saved-decoder analyses: P1 event-aligned diagnostics

[Results and plots](saved-decoder-analysis-p1-20261013T172654Z/RESULTS.md) Â· [Methods](../documentation/methods/SAVED_DECODER_ANALYSES.md)

The initial P1 analysis of **Project Idea Category 2: Additional Results from
Existing Decoders** reconstructs saved predictions for six P1 reference
models without training or new inference.
They align all 68 held-out trials to raw events, plot coordinates and signed
error over time, and summarize phase, motion, density and bias. Exact sample
matches verify a two-second trial-metadata/continuous-recording clock offset.
CNNâ€“BiLSTM errors are higher during reaching and return than preparation or the
airborne interval. Pure-window counts are sparse for short phases, and reduced
coordinates do not provide physical distance errors.

## Frozen P1 cross-participant evaluation

[Run report](frozen-p1-cross-participant-gpu-20260930T125956Z/RESULTS.md) Â· [Methods](../documentation/methods/FROZEN_P1_TRANSFER.md)

All six saved P1 models and P1 EEG/target scalers are frozen. P2â€“P12 receive no
fitting or adaptation. Primary evaluation uses weight runs 8â€“9 (45,389 windows;
fixed sandpaper throughout); all nine target runs are a separately labeled secondary
evaluation. Neural inference uses GPU, with mean/ridge on CPU.

| Model | Participant-macro RÂ² X | RÂ² Y | RÂ² Z |
| --- | ---: | ---: | ---: |
| Training mean | -0.303 | -0.535 | -0.244 |
| Ridge | -0.825 | -0.943 | -0.461 |
| CNN | -0.722 | -0.884 | -0.418 |
| CNNâ€“BiLSTM | -1.060 | -1.163 | -0.585 |
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

[Run report](six-model-comparison-p1-gpu-20260930T095506Z/RESULTS.md) Â· [Methods](../documentation/methods/SIX_MODEL_COMPARISON.md)

This comparison uses the fixed v2 P1 run split and preprocessing. Four neural
models trained on GPU; mean and ridge used CPU. Ridge regularization was selected
using validation run 7; all models use test runs 8â€“9 only for final evaluation.

| Model | Wrist X RÂ² | Wrist Y RÂ² | Wrist Z RÂ² |
| --- | ---: | ---: | ---: |
| Training mean | -0.204 | -0.324 | -0.185 |
| Ridge | 0.558 | 0.420 | 0.462 |
| Original CNN | 0.547 | 0.399 | 0.451 |
| CNNâ€“BiLSTM | 0.809 | 0.783 | 0.748 |
| CNN + ELU | 0.746 | 0.713 | 0.678 |
| EEGNet regression | 0.758 | 0.683 | 0.663 |

The ELU-only ablation improves all axes relative to the affine CNN in this run.
EEGNet regression improves over the original CNN but does not consistently exceed
the ELU-only variant. This is one participant and seed, with friction condition
3 only in test. It does not establish statistical significance or population
performance. CNNâ€“BiLSTM was retrained with a per-model seed reset, so its score
is distinct from the historical v2 run below.

## Pipeline v1: Original baseline

These are reproductions of the original notebook before preprocessing, model, or split corrections. The saved CPU and GPU runs are labeled by execution device; neither is a cross-subject evaluation.

| Run | CNN training | CNNâ€“BiLSTM training | CNNâ€“BiLSTM RÂ² (targets 1 / 2 / 3) |
|---|---:|---:|---|
| [Windows CPU](original-p1-cpu/RESULTS.md) | 259.86 s | 272.34 s | 0.833 / 0.823 / 0.770 |
| [WSL2 RTX 3060 GPU](original-p1-gpu/RESULTS.md) | 36.31 s | 153.57 s | 0.834 / 0.839 / 0.729 |

The GPU run completed training in 189.89 seconds versus 532.20 seconds on CPU (about 2.8Ã— faster in these single runs). Python/platform environments differed and model initialization was unseeded, so this is an observed timing comparison rather than a controlled benchmark.

The [baseline methods and limitations](../documentation/methods/ORIGINAL_BASELINE.md) explain window construction, architectures, the split, and why these test scores cannot yet support claims about unseen subjects. Each run page links its learning curves, target scores, and prediction scatter plot.

The public `results.json` files preserve complete numeric histories. Raw predictions, model weights, logs, and notebook execution snapshots remain local.

## Pipeline v2: Corrected Baseline

The separate [v2 notebook](../experiments/01_continuous_kinematic_decoding/corrected_baseline_v2.ipynb) fixes the first evaluation issues while retaining the two original model architectures. Its `PARTICIPANT_ID` setting selects an aligned participant; the central [changelog](../CHANGELOG.md) summarizes the version differences, and the [v2 methods](../documentation/methods/PIPELINE_V2_CORRECTED_BASELINE.md) explain each correction and its remaining limits. For the supplied nine-run participants, it trains on runs 1â€“6, validates on run 7, and tests on runs 8â€“9. EEG channels and targets are standardized from training samples only; windows never cross run boundaries.

The [first GPU result for P1](pipeline-v2-corrected-baseline-p1-gpu-20260929T173627Z/RESULTS.md) reports CNNâ€“BiLSTM RÂ² of **0.784 / 0.757 / 0.726** and CNN RÂ² of **0.547 / 0.399 / 0.451** for wrist X/Y/Z. Its run page includes learning curves, prediction plots, and scores for each held-out run separately. Both P1 test runs contain friction condition 3 only. These scores and the original random-window scores measure different evaluation settings.
