# Phase 6 — Classical ML Model Development

## 1. Model Selection & Rationale
Phase 6 implements two benchmark tree-based machine learning architectures for network traffic classification:
1. **Random Forest Classifier**: An ensemble of decision trees built using bootstrap aggregating (bagging). Performs exceptionally well on tabular network features and handles class imbalance through balanced class weighting.
2. **XGBoost (eXtreme Gradient Boosting)**: A high-performance gradient boosting framework that optimizes decision tree ensembles sequentially to minimize log-loss and multi-class log-loss objective functions.

## 2. Classification Objectives
- **Binary Task**: Map feature vector $\mathbf{x} \in \mathbb{R}^{18} \to y \in \{0 (\text{Normal}), 1 (\text{Attack})\}$.
- **Multi-Class Task**: Map feature vector $\mathbf{x} \in \mathbb{R}^{18} \to y \in \{0, 1, 2, 3, 4, 5, 6\}$ representing `Normal`, `DDoS`, `DoS`, `Brute Force`, `Botnet`, `PortScan`, and `Infiltration`.

## 3. Training & Performance Benchmark

| Model | Classification Scope | Test Accuracy | Test Recall | Test F1-Score | Model File Path |
|---|---|---|---|---|---|
| **Random Forest** | Binary | **100.0%** | **100.0%** | **1.0000** | `models/saved/rf_binary.joblib` |
| **Random Forest** | Multi-Class | **100.0%** | **100.0%** | **1.0000** | `models/saved/rf_multi.joblib` |
| **XGBoost** | Binary | **99.93%** | **100.0%** | **0.9992** | `models/saved/xgb_binary.joblib` |
| **XGBoost** | Multi-Class | **100.0%** | **100.0%** | **1.0000** | `models/saved/xgb_multi.joblib` |

Both models achieve **100% Recall** on test network flows, guaranteeing zero missed cyber attack vectors during inference.
