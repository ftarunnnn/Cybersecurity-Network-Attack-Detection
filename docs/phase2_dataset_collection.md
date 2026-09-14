# Phase 2 — Dataset Collection & Network Traffic Generator

## 1. Dataset Collection Architecture
To detect malicious network traffic automatically, our system supports standard cybersecurity benchmarks as well as a high-fidelity synthetic traffic generator matching standard schemas:
- **CIC-IDS2017**: Modern network intrusion dataset collected by the Canadian Institute for Cybersecurity.
- **UNSW-NB15**: Raw network packets created by the IXIA PerfectStorm tool in the Cyber Range Lab of UNSW Canberra.
- **CIC-DDoS2019**: Comprehensive DDoS attack dataset containing modern reflection and exploitation attack flows.

## 2. Feature Schema Definitions

| Feature Name | Type | Description |
|---|---|---|
| `src_ip` | Categorical | Source IPv4 Address |
| `dst_ip` | Categorical | Destination IPv4 Address |
| `src_port` | Numerical | Source Port Number (1024 - 65535) |
| `dst_port` | Numerical | Destination Port Number (e.g. 80, 443, 22, 21) |
| `protocol` | Categorical | Network Transport Protocol (`TCP`, `UDP`, `ICMP`) |
| `duration_sec` | Numerical | Flow duration in seconds |
| `fwd_packets` | Numerical | Total packets transmitted in forward direction |
| `bwd_packets` | Numerical | Total packets transmitted in backward direction |
| `total_packets` | Numerical | Aggregate packet count (`fwd_packets + bwd_packets`) |
| `packet_size_mean` | Numerical | Mean packet payload size (bytes) |
| `packet_size_min` | Numerical | Minimum packet size (bytes) |
| `packet_size_max` | Numerical | Maximum packet size (bytes) |
| `packet_size_std` | Numerical | Standard deviation of packet sizes |
| `flow_bytes_per_sec` | Numerical | Throughput rate in bytes per second |
| `flow_packets_per_sec` | Numerical | Throughput rate in packets per second |
| `syn_flag` | Binary | TCP SYN flag indicator (Connection initiation) |
| `ack_flag` | Binary | TCP ACK flag indicator (Acknowledgement) |
| `rst_flag` | Binary | TCP RST flag indicator (Connection reset) |
| `fin_flag` | Binary | TCP FIN flag indicator (Connection termination) |
| `psh_flag` | Binary | TCP PSH flag indicator (Push data buffer) |
| `label` | Target (Multi-class) | Class label (`Normal`, `DDoS`, `DoS`, `Brute Force`, `Botnet`, `PortScan`, `Infiltration`) |
| `is_attack` | Target (Binary) | Threat flag (`0` = Normal, `1` = Attack) |

## 3. Dataset Generation Summary
- **Training Set**: 6,000 traffic flow records (`data/raw/network_traffic.csv`)
- **Test Set**: 1,500 traffic flow records (`data/raw/network_traffic_test.csv`)
- **Class Distribution**:
  - `Normal`: 3,302 (55.0%)
  - `DDoS`: 745 (12.4%)
  - `DoS`: 594 (9.9%)
  - `Brute Force`: 476 (7.9%)
  - `Botnet`: 353 (5.9%)
  - `PortScan`: 277 (4.6%)
  - `Infiltration`: 253 (4.2%)
