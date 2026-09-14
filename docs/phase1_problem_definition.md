# Phase 1 — Problem Definition: Network Attack Detection System

## 1. Problem Overview
In modern cybersecurity infrastructure, network intrusion detection systems (NIDS) must process millions of network packets per second to differentiate between legitimate network traffic and malicious cyber attacks. Manual inspection of network logs is impossible at scale; automated machine learning and deep learning solutions are required to perform real-time attack detection, threat classification, and risk scoring.

## 2. Primary Objectives
1. **Automated Traffic Binary Classification**: Classify incoming network traffic flows as **Normal (🟢)** or **Attack (🔴)**.
2. **Multi-Class Threat Categorization**: Identify the specific attack type when malicious traffic is detected:
   - **DDoS (Distributed Denial of Service)**: Flooding attack aimed at saturating network bandwidth.
   - **DoS (Denial of Service)**: Resource exhaustion attacks (e.g. SYN Flood, Slowloris).
   - **Brute Force**: SSH / FTP / HTTP password guessing attempts.
   - **Botnet**: Malicious command-and-control (C2) bot traffic.
   - **PortScan**: Network reconnaissance and port probing.
   - **Infiltration**: Internal network compromise and unauthorized access.
3. **Prediction Confidence & Anomaly Scoring**: Provide prediction probability confidence (e.g. 96.5%) alongside an unsupervised anomaly score (reconstruction error from autoencoders) to catch zero-day attacks.

## 3. High-Priority Cybersecurity Metrics
In cybersecurity, **Recall (Sensitivity)** is the single most critical evaluation metric:
$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$
A **False Negative** (missing an actual cyber attack) can result in data breaches, ransomware deployment, or network disruption. Therefore, our preprocessing, feature selection, and model thresholding prioritize maximizing Recall while maintaining high Precision.

## 4. End-to-End System Architecture (10 Phases)

```
[Phase 2: Network Traffic Data] ──> [Phase 3: Preprocessing & Sanitation] ──> [Phase 4: EDA & Traffic Analysis]
                                                                                        │
[Phase 8: Evaluation & Benchmark] <── [Phase 6 & 7: ML + DL Models] <── [Phase 5: Feature Engineering]
           │
           └──> [Phase 9: Prediction Pipeline] ──> [Phase 10: Streamlit Dashboard + FastAPI REST Server]
```

## 5. Technology Stack & Dataset Specifications
- **Python Version**: Python 3.10+
- **Machine Learning**: Scikit-Learn (Random Forest, SelectKBest, Metrics), XGBoost
- **Deep Learning**: PyTorch (1D CNN, LSTM, Autoencoder)
- **Web & API Framework**: Streamlit (Cyberpunk Dark Mode Dashboard), FastAPI + Uvicorn
- **Data & Visualizations**: Pandas, NumPy, Plotly, Seaborn, Matplotlib
- **Standard Dataset References**: CIC-IDS2017, UNSW-NB15, CIC-DDoS2019
