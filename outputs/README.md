# Published pipeline results

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
