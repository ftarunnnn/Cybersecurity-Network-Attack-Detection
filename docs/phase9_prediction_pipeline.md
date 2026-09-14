# Phase 9 — End-to-End Prediction Pipeline

## 1. Pipeline Architectural Flow
Phase 9 encapsulates the entire intelligence pipeline into a modular inference engine (`NetworkAttackPredictor` in `src/pipeline/predictor.py`).

```
Network Packet / Flow Dict ──> [Feature Engineering] ──> [StandardScaler & Preprocessor]
                                                                  │
[Formatted Output JSON] <── [Threat Analysis & Ensembling] <── [ML / DL Model Execution]
```

## 2. Input & Output Contract

### Input JSON Structure
```json
{
  "src_ip": "192.168.1.50",
  "dst_ip": "10.0.0.5",
  "src_port": 45210,
  "dst_port": 80,
  "protocol": "TCP",
  "duration_sec": 0.01,
  "fwd_packets": 2000,
  "bwd_packets": 2,
  "total_packets": 2002,
  "packet_size_mean": 64.0,
  "packet_size_min": 40.0,
  "packet_size_max": 128.0,
  "packet_size_std": 5.0,
  "flow_bytes_per_sec": 1281280.0,
  "flow_packets_per_sec": 200200.0,
  "syn_flag": 1,
  "ack_flag": 0,
  "rst_flag": 0,
  "fin_flag": 0,
  "psh_flag": 0
}
```

### Output JSON Structure
```json
{
  "status": "🔴 Attack",
  "is_attack": true,
  "attack_type": "DDoS",
  "confidence_percent": 100.0,
  "anomaly_score": 7.2129,
  "is_zero_day_anomaly": false,
  "flow_info": {
    "src_ip": "192.168.1.50",
    "dst_ip": "10.0.0.5",
    "dst_port": 80,
    "protocol": "TCP"
  }
}
```
