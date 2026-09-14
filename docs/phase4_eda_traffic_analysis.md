# Phase 4 — EDA & Network Traffic Analysis

## 1. Overview & Objectives
Exploratory Data Analysis (EDA) investigates key distributional differences between Normal network traffic and Cyber Attack vectors to guide feature engineering and model optimization.

## 2. Key Statistical Discoveries

### 1. Normal vs Attack Distribution
- Total Records Analyzed: **6,000**
- Normal Traffic: **3,302 (55.03%)**
- Malicious Attack Traffic: **2,698 (44.97%)**

### 2. Attack-Type Breakdown
- **Normal**: 3,302
- **DDoS**: 745 (High packet volume, small packet payloads, high SYN flags)
- **DoS**: 594 (Extended flow duration, HTTP/TCP connection exhaustion)
- **Brute Force**: 476 (Repeated connection attempts on SSH/FTP/RDP ports 22, 21, 3389)
- **Botnet**: 353 (Periodic small-packet Command & Control signals)
- **PortScan**: 277 (Extremely short duration, single-packet SYN probes across port ranges)
- **Infiltration**: 253 (Large payload transfers, anomalous TCP RST/FIN flag combinations)

### 3. Metric Comparison: Normal vs Attack Means

| Feature Metric | Normal Mean | Attack Mean | Key Diagnostic Insight |
|---|---|---|---|
| `duration_sec` | 2.57 s | 13.42 s | Attacks often exhibit extreme short probes (PortScan) or persistent holds (DoS/Botnet) |
| `fwd_packets` | 50.39 | 935.59 | Massive forward packet burst characteristic of DDoS floods |
| `bwd_packets` | 60.47 | 25.07 | Asymmetric packet flow in attack traffic (server responses blocked or ignored) |
| `packet_size_mean` | 511.56 B | 202.53 B | Malicious traffic frequently uses header-only or small payload probes |

## 3. Visualization Pipeline
The EDA system powers 4 dynamic Plotly chart components in the Streamlit Dashboard:
1. **Histogram**: Comparative distribution of `packet_size_mean` and `duration_sec`.
2. **Box Plot**: Outlier analysis across flow rate metrics (`flow_bytes_per_sec`, `flow_packets_per_sec`).
3. **Correlation Heatmap**: Pearson feature correlation matrix identifying collinear pairs.
4. **Count Plot**: Multi-class attack imbalance visualization.
