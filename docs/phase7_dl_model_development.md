# Phase 7 — Deep Learning Model Development (PyTorch)

## 1. Overview & Architectural Design
Phase 7 implements 3 PyTorch deep neural network architectures to detect spatial feature correlations, temporal behavior sequences, and novel zero-day network anomalies.

## 2. Deep Learning Architectures

### 1. 1D CNN (Convolutional Neural Network)
- **Layer Design**: 2 x 1D Convolutional Blocks (`Conv1d(32)` + `BatchNorm1d` + `ReLU` $\to$ `Conv1d(64)` + `BatchNorm1d` + `ReLU`) $\to$ `AdaptiveAvgPool1d(4)` $\to$ `Dropout(0.3)` $\to$ Fully Connected Classifier.
- **Role**: Learns spatial interactions across packet flow metrics (e.g. relationship between `packet_size_mean`, `syn_flag`, and `flow_bytes_per_sec`).

### 2. LSTM (Long Short-Term Memory Network)
- **Layer Design**: 2-layer stacked LSTM (`input_size=18`, `hidden_size=32`, `dropout=0.2`) $\to$ Linear Output Layer.
- **Role**: Processes sequential flow vectors to capture temporal network behavior patterns typical of persistent DoS connection holds and Botnet beacons.

### 3. Autoencoder (Unsupervised Anomaly Detector)
- **Layer Design**: Encoder (`18 -> 16 -> 4`) $\to$ Decoder (`4 -> 16 -> 18`).
- **Training Strategy**: Trained exclusively on **Normal** network traffic.
- **Anomaly Scoring**: Calculates Mean Squared Error (MSE) reconstruction loss:
$$\mathcal{L}_{\text{recon}} = \frac{1}{d} \sum_{i=1}^d (x_i - \hat{x}_i)^2$$
- **Dynamic Thresholding**: Anomaly threshold set automatically at the **95th percentile** of normal traffic reconstruction loss ($\text{Threshold} = 0.4785$). Any flow exceeding this error threshold is flagged as an unclassified zero-day anomaly.

## 3. Performance Summary

| Architecture | Classification Type | Test Accuracy | Test Recall | PyTorch Artifact Path |
|---|---|---|---|---|
| **1D CNN** | Multi-Class | **100.0%** | **100.0%** | `models/saved/cnn_model.pt` |
| **LSTM** | Multi-Class | **99.87%** | **99.87%** | `models/saved/lstm_model.pt` |
| **Autoencoder** | Unsupervised Anomaly | **94.47%** | **94.58%** | `models/saved/autoencoder_model.pt` |
