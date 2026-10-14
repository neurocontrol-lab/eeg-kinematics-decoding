# P1 saved-model inference benchmark

Measured 14 October 2026. Six saved P1 v2 references from `six-model-comparison-p1-gpu-20260930T095506Z`, not the historical v1 checkpoints. No training or adaptation.

## One-window latency

Times are milliseconds, median / 95th percentile. Lower is faster.

| Model | CPU ms | GPU ms | CPU median / GPU median |
|---|---:|---:|---:|
| Training mean | 0.003 / 0.005 | CPU baseline | — |
| Ridge | 0.004 / 0.006 | CPU baseline | — |
| CNN | 1.114 / 1.330 | 1.402 / 1.599 | 0.79× |
| CNN–BiLSTM | 4.993 / 6.847 | 4.306 / 5.069 | 1.16× |
| CNN + ELU | 1.064 / 1.269 | 1.418 / 1.748 | 0.75× |
| EEGNet regression | 1.127 / 1.356 | 1.468 / 1.839 | 0.77× |

GPU ratio below 1 means the GPU is slower. Mean and ridge retain their existing NumPy CPU implementations; GPU versions were not measured.

## Batches of 64

These times are per batch, not single-window response latencies. Waiting to accumulate a batch is excluded.

| Model | CPU median ms/batch | GPU median ms/batch | CPU / GPU |
|---|---:|---:|---:|
| Training mean | 0.0034 | — | — |
| Ridge | 0.0556 | — | — |
| CNN | 5.3847 | 2.0378 | 2.64× |
| CNN–BiLSTM | 19.7498 | 5.2570 | 3.76× |
| CNN + ELU | 5.5231 | 2.1676 | 2.55× |
| EEGNet regression | 5.4108 | 2.1044 | 2.57× |

## Method

- Intel Core i7-11800H and RTX 3060 Laptop GPU (6 GB), same WSL2 TensorFlow 2.20.0 environment. Four intra-op/BLAS threads, one inter-op thread. CPU and GPU run sequentially in separate processes; CPU process hides CUDA.
- Identical 3,642 P1 held-out run 8–9 windows, 128 samples × 32 EEG channels, float32, standardized with saved P1 training scaler. Input and target-index hashes match across devices.
- Warm-up: 20 calls per model/batch shape, including tracing. Three rounds of 100 measured calls per model and batch size (300 total). Seeded randomized model order within each round; inputs vary through the same window set.
- Neural path: compiled `tf.function(model(x, training=False))`, not high-overhead `model.predict`. Timed boundary starts with an already-prepared NumPy window and ends with a completed NumPy XYZ output. CPU tensor conversion, GPU transfer, dispatch and output synchronization are included.
- Baselines use saved NumPy broadcast-and-copy and ridge matrix multiplication. Different implementations and precisions are preserved as stored.
- Acquisition, filtering, scaling, window construction, model loading, initial tracing, and inverse target scaling are excluded. Cold-start and complete closed-loop latency are not measured.
- 95th percentile describes these 300 observed calls, not a confidence bound. Laptop temperature, power state and background activity were not controlled. Round medians remain in device JSON files.

## Interpretation

For single windows, the CPU is faster for the three CNN variants. GPU acceleration gives CNN–BiLSTM only a modest latency reduction. For batches of 64, all neural models are faster on GPU (2.55–3.76×).

Every measured one-window p95 is below 128 ms, the current stride at 500 Hz. This does not demonstrate end-to-end real-time operation. A 128-sample input spans 256 ms of data, and a 128-ms stride updates at about 7.8 Hz. Batch throughput is not streaming latency.

## Verification and artifacts

Source hashes and all in-memory neural weights remained unchanged. The first 64 predictions match the saved predictions within declared replay tolerances (absolute 0.01, relative 0.001, mean absolute difference below 0.001); maximum observed difference is 0.0005068182945251465. This is a numerical replay check, not accuracy reevaluation.

`cpu_results.json` and `gpu_results.json` retain hardware/software, source hashes, raw-input hashes, settings, round summaries and replay checks. Per-model/batch `*_latency_ms.npy` files retain raw timings. `experiment_source.py` snapshots the script. Reproduce using `experiments/benchmark_inference.py --repo <repo> --output <new-output> --device cpu` and then `--device gpu`.

The published device JSON uses a repository-relative source path instead of the
local absolute path. Numeric measurements and the executed-source snapshot are
unchanged. Commit preparation checks all 20 timing arrays (6,000 calls) against
the reported median and p95, confirms both device runs completed, and compares
their input and target-index hashes.
