# EEG-Based Hand Motion Prediction

## Project Overview
This project uses EEG (electroencephalogram) data to predict hand motion. It demonstrates how brain signals can be processed and used to predict physical movements, which has applications in brain-computer interfaces, assistive technologies, and rehabilitation.

## Dataset
The project uses the WAY-EEG-GAL dataset, which contains:
- EEG recordings from multiple subjects
- Position recordings from trackers on the object, index finger, thumb, and wrist
- Various session recordings for different participants

## Implementation
The implementation includes:

1. **Data Loading and Preprocessing**:
   - Loading EEG and kinematic data from .mat files
   - Normalizing data to a [-1, 1] range
   - Creating windowed samples for time-series prediction
   - Train-test splitting for model evaluation

2. **Model Architecture**:
   - CNN-based model with temporal and spatial filters
   - Depthwise and separable convolutions for efficient processing
   - Batch normalization and dropout for regularization
   - Dense output layer for predicting hand motion coordinates

3. **Alternative Models**:
   - LSTM/Bidirectional LSTM model for sequence processing
   - Various architectural experiments for performance comparison

4. **Evaluation**:
   - Mean Squared Error (MSE) and Mean Absolute Error (MAE) metrics
   - Visualization of predicted vs. actual hand movements

## Requirements
- Python 3.x
- TensorFlow/Keras
- NumPy
- SciPy
- Matplotlib
- h5py
- MNE (for EEG processing)

## Usage
The [original pipeline](pipeline_v1_original.ipynb) reproduces the fork's baseline. The [corrected baseline](pipeline_v2_corrected_baseline.ipynb) is the current starting point for new experiments. To run either notebook:

1. Ensure all dependencies are installed
2. Download the WAY-EEG-GAL dataset from Kaggle
3. Run the notebook cells sequentially

## Results
The [original P1 baseline](documentation/ORIGINAL_BASELINE.md) explains the data windows, both models, and evaluation limitations. The [published CPU and GPU runs](outputs/README.md) include per-target metrics, complete training histories, and plots. The strongest CNN–BiLSTM run reached R² of 0.834, 0.839, and 0.729 on the three wrist-position targets (X/Y/Z). These are random overlapping-window test scores on one participant, not evidence of generalization to new subjects or recording series.

The reduced `kin_P*.mat` matrices have three columns. [Direct comparison with the original P1 and P6 recordings](documentation/KINEMATICS_PROVENANCE.md) identifies them as the wrist tracker's X, Y, and Z position channels. The reduced values span 0–1; their exact filtering and scaling recipe is not supplied by the reduced dataset. The v1 notebook rescales these values to [-1, 1]; v2 standardizes them using training runs only.

The [v2 corrected baseline](pipeline_v2_corrected_baseline.ipynb) has a `PARTICIPANT_ID` setting near the top for aligned participants P1–P12. The central [changelog](CHANGELOG.md) records the differences between versions, and the [v2 methods](documentation/PIPELINE_V2_CORRECTED_BASELINE.md) explain how each correction works. Its [first GPU result for P1](outputs/pipeline-v2-corrected-baseline-p1-gpu-20260929T173627Z/RESULTS.md) holds out entire recording runs and fits EEG and wrist-position scalers on training data only. The v1 model definitions and saved runs are preserved.

## Future Work

- Experiment with more advanced architectures
- Incorporate attention mechanisms
- Test on real-time EEG data
- Expand to more complex movement predictions
