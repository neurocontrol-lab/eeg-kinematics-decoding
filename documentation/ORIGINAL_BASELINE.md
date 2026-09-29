# Original notebook baseline

The first experiment ran the original P1 models before any scientific corrections.
The completed [CPU](../outputs/original-p1-cpu/RESULTS.md) and
[GPU](../outputs/original-p1-gpu/RESULTS.md) runs publish metrics, full numeric
histories and plots under `outputs/`. Model files, raw prediction arrays,
training logs and notebook snapshots remain local.
Result saving now lives directly in the notebook; the separate runner was removed.
Only logging and timing were added. Model, preprocessing and split code are unchanged.

## Run locally

Open `hand_motion.ipynb` in Jupyter or VS Code and select the existing `venv`
Python kernel. Save the notebook first. With local data already downloaded,
skip the installation/download cells and start at **Run and save the original
P1 baseline**. Run in order through **Save CNN–BiLSTM results**.

The setup cell creates a timestamped, CPU/GPU-labeled directory under `outputs/`.
The two save cells
record each fitted model, predictions, epoch histories, runtime, MSE, MAE and R²
in that directory. Environment versions and an on-disk notebook snapshot are
also recorded. Training logs and plots stay in the notebook: save it after
execution to retain them. There is no separate logging script to run.

The later Colab file-transfer and P2 demonstration cells are separate from the
P1 training experiment. Avoid rerunning setup midway through a run, since it
starts a new results directory.

The original CPU environment is Python 3.9.13 / TensorFlow 2.20.0 on native Windows.
It detects the CPU only, although NVIDIA's driver detects an RTX 3060.
TensorFlow stopped native-Windows CUDA support after 2.10. GPU execution with a
modern release requires another supported environment, such as WSL2 or Colab.
Source: https://www.tensorflow.org/install/pip

## Run on the local GPU (WSL2)

A separate Ubuntu WSL2 environment contains Python 3.12.3, TensorFlow 2.20.0
with CUDA dependencies and Keras 3.10.0. A 1024 × 1024 matrix
multiplication was verified on `/device:GPU:0`, the RTX 3060 Laptop GPU.

With WSL2 and a compatible NVIDIA driver installed, create and launch a GPU
environment from an Ubuntu terminal in the repository directory:

```bash
python3 -m venv ~/.venvs/eeg-tf-gpu
source ~/.venvs/eeg-tf-gpu/bin/activate
python -m pip install 'tensorflow[and-cuda]==2.20.0' 'keras==3.10.0' 'numpy==2.0.2' 'scipy==1.13.1' 'scikit-learn==1.6.1' matplotlib ipykernel jupyterlab
python -m ipykernel install --user --name eeg-tf-gpu --display-name 'Python (EEG TensorFlow GPU)'
TF_FORCE_GPU_ALLOW_GROWTH=true jupyter lab --no-browser
```

Open the local Jupyter URL printed in the terminal, open `hand_motion.ipynb`,
and select **Python (EEG TensorFlow GPU)**. Follow the P1 run instructions above.
Memory growth lets TensorFlow allocate GPU memory as needed. The GPU kernel
runs inside WSL; selecting the existing native Windows kernel still uses CPU.
No additional notebook runner script is needed.

The original CNN took 36.31 seconds for 15 epochs (CPU: 259.86 seconds);
CNN–BiLSTM took 153.57 seconds for 20 epochs (CPU: 272.34 seconds).
Total training was 189.89 seconds versus 532.20 seconds, about 2.8× faster
in these single runs. These timings include validation and initial compilation.
Different Python/platform environments and unseeded training mean this is not
a controlled benchmark. GPU execution does not itself imply better accuracy.
Both models' test targets exactly match the CPU run. Public `results.json` files
contain the complete numeric histories; the detailed run pages show the plots.

| Device | Model | Target 1 R² | Target 2 R² | Target 3 R² |
|---|---|---:|---:|---:|
| CPU | CNN | 0.6730 | 0.6427 | 0.4899 |
| GPU | CNN | 0.6842 | 0.6517 | 0.5932 |
| CPU | CNN–BiLSTM | 0.8334 | 0.8232 | 0.7696 |
| GPU | CNN–BiLSTM | 0.8343 | 0.8390 | 0.7291 |

The [CPU learning curves](../outputs/original-p1-cpu/learning_curves.png) and
[GPU learning curves](../outputs/original-p1-gpu/learning_curves.png) show the
training and validation/test behavior at every epoch. The
[GPU observed-versus-predicted plot](../outputs/original-p1-gpu/prediction_scatter.png)
shows the spread behind the R² scores. Test windows are randomly ordered, so
their predictions should not be connected as a movement trajectory.

## Input and target

The original training uses P1 only. `eeg` loads as `(32, 1538345)` and is
transposed to `(1538345, 32)`. `kin` is `(1538345, 3)`. Each array is scaled with
one global minimum and maximum to [-1, 1]. The scaling is not per EEG channel.

For each start index `i`, the input is `eeg[i:i+128]` and the target is
`kin[i+128]`. Thus the last EEG input is at `i+127`; the target is one sample
after it. Starts advance by 64 samples. This generates 24,035 windows.

The paper gives 500 Hz for the original EEG and kinematics, making a 128-sample
window 256 ms and the stride 128 ms. The cleaned files lack sampling-rate
metadata; the notebook's 512 Hz comment is not independent evidence.

The three target columns have not yet been traced to named original channels.
They are called target 1, 2 and 3 in this documentation. The notebook's finger
and glove labels should not be treated as established sensor identities.

## Model 1: CNN

Shapes below omit the batch dimension.

| Layer | Output shape | Meaning |
|---|---|---|
| Input | 128 × 32 × 1 | Time × electrode × feature map |
| Conv2D, 8 filters, kernel (64,1), same padding | 128 × 32 × 8 | Eight temporal filters applied separately at each electrode |
| Batch normalization | 128 × 32 × 8 | Normalize feature maps during training; use moving statistics at inference |
| DepthwiseConv2D, kernel (1,32), multiplier 2 | 128 × 1 × 16 | Two spatial electrode combinations for each of eight temporal features |
| Batch normalization, dropout 0.25 | 128 × 1 × 16 | Normalization and regularization |
| SeparableConv2D, 16 filters, kernel (16,1), same padding | 128 × 1 × 16 | Depthwise temporal filtering followed by pointwise feature mixing |
| Batch normalization | 128 × 1 × 16 | Feature normalization |
| AveragePooling2D, (4,1) | 32 × 1 × 16 | Average groups of four time steps |
| Dropout 0.25, flatten | 512 | Turn feature maps into one vector |
| Dense(3) | 3 | Continuous regression outputs |

**Important:** the original CNN has no nonlinear activations. At inference,
batch normalization is affine, dropout is disabled, and all remaining operations
are linear/affine. Its complete inference function is therefore affine in its
input window. Factorization constrains the mapping; training dynamics still
differ from ordinary linear regression because of batch normalization and
dropout. We do not add activations in this reproduction.

## Model 2: CNN–BiLSTM

| Layer | Output shape | Meaning |
|---|---|---|
| Input | 128 × 32 | Time × EEG channels |
| Conv1D, 32 filters, kernel 3, ReLU, same padding | 128 × 32 | Local patterns combining all electrodes |
| Batch normalization, dropout 0.3 | 128 × 32 | Normalize and regularize features |
| Bidirectional LSTM(64), return_sequences=False | 128 | Concatenate 64-unit forward and backward summaries |
| Dropout 0.3 | 128 | Regularization |
| Dense(64), ReLU | 64 | Nonlinear combination of sequence features |
| Dropout 0.2 | 64 | Regularization |
| Dense(3) | 3 | Continuous regression outputs |

An LSTM maintains a memory state while reading a sequence. Learned forget,
input and output gates control which information is retained, updated and
exposed. The bidirectional wrapper runs separate LSTMs in opposite directions.
Both read only the observed 128-sample window; neither sees EEG after the
prediction target. The default is not stateful, so memory does not persist
between windows. Source: https://keras.io/api/layers/recurrent_layers/bidirectional/

The first CNN isolates temporal filtering from electrode mixing. In the second
model, Conv1D combines electrodes immediately, and the LSTM models the ordering
of the resulting features. There is no attention mechanism in either model.
The second architecture replaces the first `model` variable and trains from
scratch; it does not fine-tune the first CNN's weights.

## Training and evaluation

Both models use Adam with its installed Keras defaults, MSE as the training
objective, MAE as an additional metric, and batches of 64. The CNN runs for 15
epochs; the CNN–BiLSTM runs for 20. Training shuffles windows by default.

The split is 80% training and 20% test with `random_state=42`, giving 19,228 and
4,807 windows respectively. The same test set is passed to `validation_data`
at every epoch. No separate validation set, early stopping or best-epoch
selection is present. The final epoch's weights are evaluated and saved.

Only the split has a fixed random seed. The original model initialization and
training shuffle are unseeded, so reruns need not match numerically.

- MSE: average squared prediction error; larger errors receive more weight.
- MAE: average absolute prediction error in normalized target units.
- R²: `1 - sum((y-pred)^2) / sum((y-mean(y))^2)`, per target. One is perfect;
  zero equals the test-mean reference; negative values are worse.

Training loss includes dropout behavior; validation uses inference behavior.
This is one reason validation loss can be below training loss.

## Issues recorded for later, not corrected now

1. Overlapping windows are randomly split, so partitions can share EEG samples.
2. Normalization uses all P1 samples before splitting.
3. Windows can span concatenated recording-series boundaries.
4. The test set is also used as validation during training.
5. The three targets' original channel names and units remain unverified.
6. The CNN has no nonlinear activations.
7. Later P2 demonstration cells assume incorrect EEG orientation, use 64 rather
   than 128 samples, and use different scaling. They are not part of this P1 run.
8. The plotted first test windows are randomly ordered; connecting them does
   not show a continuous movement trajectory.

These scores are an original-pipeline reproduction reference, not an estimate
of independent-series or unseen-subject generalization. Corrections belong in
a separate experiment after reviewing this run.

## Optional Colab execution

If local training becomes inconvenient, open `hand_motion.ipynb` in Colab.
Place `way_eeg_clean/eeg_P1.mat` and `way_eeg_clean/kin_P1.mat` under the current
working directory, select a GPU runtime, install dependencies as needed, and
run the same setup and P1 cells through **Save CNN–BiLSTM results**. The setup
also works without a local notebook file, but then cannot save a source snapshot.
Download the results directory before the Colab runtime is discarded.

Record Colab's package versions and device as a distinct run; differing Keras
versions, hardware and unseeded initialization can change the results. This
task does not start a cloud runtime or upload EEG recordings automatically.
