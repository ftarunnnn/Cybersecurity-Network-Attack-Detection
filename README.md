# 🔐 Cybersecurity — Network Attack Detection System (10-Phase Pipeline)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosted%20Trees-1185C7.svg)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688.svg)](https://fastapi.tiangolo.com/)

An end-to-end industrial-grade **Network Attack Detection & Threat Intelligence System** capable of real-time network traffic classification, multi-class attack categorization, anomaly scoring, and live monitoring.

---

## 📌 Project Architecture & 10-Phase Roadmap

- [x] **Phase 1 — Problem Definition**: Dual classification (Normal vs Attack), multi-class attack labels (DDoS, DoS, Brute Force, Botnet, PortScan, Infiltration), confidence & anomaly scoring specs.
- [ ] **Phase 2 — Dataset Collection**: Network traffic dataset generator modeling CIC-IDS2017, UNSW-NB15, & CIC-DDoS2019 schema formats + custom CSV support.
- [ ] **Phase 3 — Data Preprocessing**: Deduplication, null handling, IP/Port/Protocol encoding, and Robust MinMax scaling.
- [ ] **Phase 4 — EDA & Traffic Analysis**: Statistical summaries, attack distribution, correlation heatmaps, box plots, and class imbalance metrics.
- [ ] **Phase 5 — Feature Engineering**: Derived flow rate features (packets/sec, bytes/sec, SYN/ACK ratios) & Random Forest Gini feature selection.
- [ ] **Phase 6 — ML Model Development**: Classical ML models (Random Forest & XGBoost classifiers).
- [ ] **Phase 7 — DL Model Development**: PyTorch Deep Learning suite featuring 1D CNN, LSTM, and Autoencoder (Unsupervised Anomaly Detection).
- [ ] **Phase 8 — Model Evaluation**: High-recall optimization, confusion matrices, multi-class ROC-AUC curves, and metrics reporting.
- [ ] **Phase 9 — Prediction Pipeline**: Unified modular inference engine returning threat status (🟢 Normal / 🔴 Attack), confidence score, and anomaly level.
- [ ] **Phase 10 — Deployment & Monitoring**: FastAPI REST server + Streamlit Cybersecurity Dashboard with live traffic simulator and packet inspector.

---

## 📂 Repository Structure

```
Cybersecurity-Network-Attack-Detection/
├── docs/                                # Phase 1 - 10 Technical Documentation
│   ├── phase1_problem_definition.md
│   ├── phase2_dataset_collection.md
│   ├── phase3_data_preprocessing.md
│   ├── phase4_eda_traffic_analysis.md
│   ├── phase5_feature_engineering.md
│   ├── phase6_ml_model_development.md
│   ├── phase7_dl_model_development.md
│   ├── phase8_model_evaluation.md
│   ├── phase9_prediction_pipeline.md
│   └── phase10_deployment_monitoring.md
├── src/                                 # Core Source Code
│   ├── data/                           # Phase 2: Data Generator & Loader
│   ├── preprocessing/                  # Phase 3 & 5: Preprocessor & Feature Engineering
│   ├── analysis/                       # Phase 4: EDA & Plotly Visualizers
│   ├── models/                         # Phase 6 & 7: ML (RF, XGB) & DL (CNN, LSTM, Autoencoder)
│   ├── evaluation/                     # Phase 8: Model Evaluator
│   ├── pipeline/                       # Phase 9: Unified Predictor Pipeline
│   └── api/                            # Phase 10: FastAPI REST Endpoints
├── tests/                              # Automated Pytest Suite
├── reports/                            # Generated Metrics & EDA JSON Summaries
├── app.py                              # Phase 10: Streamlit Cybersecurity Monitoring Dashboard
├── requirements.txt                    # Project Dependencies
└── README.md                           # Project Documentation
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/ftarunnnn/Cybersecurity-Network-Attack-Detection.git
cd Cybersecurity-Network-Attack-Detection
pip install -r requirements.txt
```

### 2. Generate Dataset & Run Pipeline
```bash
python data/generate_network_data.py
python -m src.models.train_ml_models
python -m src.models.train_dl_models
```

### 3. Launch Streamlit Cybersecurity Dashboard
```bash
streamlit run app.py
```

### 4. Launch FastAPI REST Server
```bash
uvicorn src.api.main:app --port 8000 --reload
```