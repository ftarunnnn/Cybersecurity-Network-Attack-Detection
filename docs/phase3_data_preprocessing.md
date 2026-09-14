# Phase 3 — Data Preprocessing & Cleaning

## 1. Objective & Scope
The Phase 3 Data Preprocessing engine sanitizes raw network traffic logs to ensure numerical stability and format consistency before feature engineering and model training.

## 2. Preprocessing Steps

### 1. Deduplication
Raw network logs frequently record duplicated packet flows from multi-interface captures. All identical rows across feature columns are identified and removed.

### 2. Missing & Infinity Sanitation
Network throughput calculations (`flow_bytes_per_sec`, `flow_packets_per_sec`) can produce infinite values when `duration_sec == 0`.
- Infinite values (`+inf`, `-inf`) are converted to `NaN`.
- `NaN` values are imputed with median feature values calculated during training.

### 3. Categorical Feature Encoding
- **Transport Protocol**: Mapped to ordinal integer codes (`TCP: 0`, `UDP: 1`, `ICMP: 2`).
- **Attack Labels**: Label-encoded into numerical indices (`0` to `6`) matching the 7 target classes (`Normal`, `DDoS`, `DoS`, `Brute Force`, `Botnet`, `PortScan`, `Infiltration`).

### 4. Numerical Standardization
To ensure equal feature weighting for distance-based ML models and gradient descent in neural networks, numerical features are standardized using Z-score normalization:
$$z = \frac{x - \mu}{\sigma}$$
- Means ($\mu$) and standard deviations ($\sigma$) are fitted exclusively on the training set and stored in `models/saved/preprocessor.joblib`.
