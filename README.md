# EEG-Based Hand Motion Prediction

## Project Overview
This project uses EEG (electroencephalogram) data to predict hand motion. It demonstrates how brain signals can be processed and used to predict physical movements, which has applications in brain-computer interfaces, assistive technologies, and rehabilitation.

## Dataset
The project uses the WAY-EEG-GAL dataset, which contains:
- EEG recordings from multiple subjects
- Kinematic data from a data glove that tracks hand movements
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
The main implementation is in the Jupyter notebook `hand_motion.ipynb`. To run the project:

1. Ensure all dependencies are installed
2. Download the WAY-EEG-GAL dataset from Kaggle
3. Run the notebook cells sequentially

## Results
The model demonstrates the ability to predict hand movements from EEG signals with reasonable accuracy, showing the potential for brain-computer interface applications.

## Future Work
- Experiment with more advanced architectures
- Incorporate attention mechanisms
- Test on real-time EEG data
- Expand to more complex movement predictions