# Phase 10 — Deployment & Interactive Monitoring Dashboard

## 1. System Overview
Phase 10 deploys the Network Attack Detection system using **FastAPI** for low-latency REST API inference and **Streamlit** for an interactive cybersecurity monitoring dashboard.

```
Incoming Telemetry / Packets
        │
        ├──> [FastAPI REST Endpoint: /predict] ──> JSON Threat Response
        │
        └──> [Streamlit Cyberpunk Dashboard]
                    ├── 📊 EDA & Traffic Analysis Hub
                    ├── ⚡ Live Traffic Stream Simulator
                    ├── 🎯 Single Packet & Batch Inspector
                    ├── 🧠 Model Studio & Benchmarks
                    └── 🔌 API Explorer & Service Status
```

## 2. FastAPI Endpoints & Contract
- `GET /health`: Health probe returning service readiness and model status.
- `POST /predict`: Single packet classification returning status (`🟢 Normal` / `🔴 Attack`), attack type (`DDoS`, `DoS`, `Brute Force`, `Botnet`, `PortScan`, `Infiltration`), confidence score (%), and anomaly score.
- `POST /predict/batch`: Bulk CSV network log packet classification.
- `GET /simulate`: Generates a real-time sample packet and runs instant threat prediction.

## 3. Interactive Streamlit Dashboard Features
- **Live Threat Telemetry**: Instant alert feed displaying source IP, destination port, threat confidence, and autoencoder reconstruction anomaly error.
- **Single & Batch Inspector**: Interactive parameter inputs and CSV drag-and-drop file processing with CSV export.
- **Model Benchmark Studio**: Visual confusion matrices and side-by-side recall bar charts comparing Random Forest, XGBoost, 1D CNN, LSTM, and Autoencoders.

## 4. Verification & Testing
Run pytest to verify the full pipeline:
```bash
pytest tests/test_pipeline.py -v
```
