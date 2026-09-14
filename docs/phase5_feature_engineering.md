# Phase 5 — Feature Engineering & Selection

## 1. Objectives
Phase 5 derives advanced network domain features to capture attack characteristics that raw features (such as raw packet counts) fail to highlight.

## 2. Derived Feature Specifications

### 1. `derived_packets_per_sec`
$$\text{Packets/sec} = \frac{\text{total\_packets}}{\text{duration\_sec} + \epsilon}$$
- **Cybersecurity Rationale**: Extremely high throughput rate is an immediate indicator of volumetric **DDoS** and **SYN flood** attacks.

### 2. `derived_bytes_per_sec`
$$\text{Bytes/sec} = \frac{\text{total\_packets} \times \text{packet\_size\_mean}}{\text{duration\_sec} + \epsilon}$$
- **Cybersecurity Rationale**: Measures network bandwidth saturation during exfiltration or data flooding.

### 3. `derived_avg_packet_size`
$$\text{Avg Size} = \frac{\text{packet\_size\_min} + \text{packet\_size\_max} + \text{packet\_size\_mean}}{3}$$
- **Cybersecurity Rationale**: Distinguishes small control ping attacks from large data exfiltration (Infiltration).

### 4. `derived_syn_ack_ratio`
$$\text{SYN/ACK Ratio} = \frac{\text{syn\_flag} + \epsilon}{\text{ack\_flag} + \epsilon}$$
- **Cybersecurity Rationale**: High SYN/ACK ratios signal incomplete handshake attempts common in **SYN Floods** and **PortScans**.

### 5. `derived_fwd_bwd_ratio`
$$\text{Fwd/Bwd Ratio} = \frac{\text{fwd\_packets} + \epsilon}{\text{bwd\_packets} + \epsilon}$$
- **Cybersecurity Rationale**: Normal web browsing exhibits balanced bidirectional traffic; DDoS floods show heavily asymmetric forward traffic.

## 3. Feature Selection Methodology
Random Forest Gini Impurity reduction evaluates feature importance scores across all original and derived features. Top features are selected for downstream ML & DL models to reduce dimensionality and improve inference speed.
