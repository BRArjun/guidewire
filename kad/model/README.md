# Anomaly Detection Models

## Overview
This contains implementations of various anomaly detection models. These models are designed to detect anomalies in time-series data and are implemented using different machine learning techniques.

## Models
The following models are included in this repository:

### 1. **Autoencoder Model** (`autoencoder_model.py`)
- Implements an LSTM-based Autoencoder for anomaly detection.
- Learns to reconstruct normal sequences and identifies anomalies as deviations from expected patterns.
- Uses Mean Squared Error (MSE) loss for training.
- Implements early stopping to avoid overfitting.

### 2. **Hidden Markov Model (HMM)** (`hmm_model.py`)
- Uses a Gaussian Hidden Markov Model for anomaly detection.
- Learns the probability distribution of the normal sequences.
- Uses residual errors to determine anomalies.

### 3. **LSTM Model** (`lstm_model.py`)
- Implements an LSTM-based time-series forecasting model.
- Uses two LSTM layers followed by a Dense layer for prediction.
- Identifies anomalies by computing the forecast error threshold.
- Uses early stopping for training optimization.

### 4. **SARIMA Model** (`sarima_model.py`)
- Implements a Seasonal ARIMA (SARIMA) model for anomaly detection.
- Forecasts future values and compares them with actual observations.
- Uses a dynamically computed error threshold for anomaly identification.

### 5. **Base Model Interface** (`i_model.py`)
- Defines an abstract base class `IModel` for implementing anomaly detection models.
- Enforces the implementation of `train()` and `test()` methods.
- Contains `ModelException` class for error handling.

### 6. **Model Utility Functions** (`model_utils.py`)
- Provides helper functions to initialize models based on names.
- Supports dynamic selection of models such as Autoencoder, LSTM, HMM, and SARIMA.

## Dependencies
Ensure you have the following dependencies installed before running the models:

```bash
pip install numpy pandas scikit-learn keras tensorflow statsmodels hmmlearn matplotlib
```

