# Published original P1 baseline runs

These are reproductions of the original notebook before preprocessing, model, or split corrections. The saved CPU and GPU runs are labeled by execution device; neither is a cross-subject evaluation.

| Run | CNN training | CNN–BiLSTM training | CNN–BiLSTM R² (targets 1 / 2 / 3) |
|---|---:|---:|---|
| [Windows CPU](original-p1-cpu/RESULTS.md) | 259.86 s | 272.34 s | 0.833 / 0.823 / 0.770 |
| [WSL2 RTX 3060 GPU](original-p1-gpu/RESULTS.md) | 36.31 s | 153.57 s | 0.834 / 0.839 / 0.729 |

The GPU run completed training in 189.89 seconds versus 532.20 seconds on CPU (about 2.8× faster in these single runs). Python/platform environments differed and model initialization was unseeded, so this is an observed timing comparison rather than a controlled benchmark.

The [baseline methods and limitations](../documentation/ORIGINAL_BASELINE.md) explain window construction, architectures, the split, and why these test scores cannot yet support claims about unseen subjects. Each run page links its learning curves, target scores, and prediction scatter plot.

The public `results.json` files preserve complete numeric histories. Raw predictions, model weights, logs, and notebook execution snapshots remain local.
