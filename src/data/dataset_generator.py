import os
import numpy as np
import pandas as pd

class NetworkDatasetGenerator:
    """
    Generates realistic network traffic datasets matching standard schemas
    such as CIC-IDS2017, UNSW-NB15, and CIC-DDoS2019.
    
    Includes multi-class attack categories:
    - Normal (🟢)
    - DDoS (🔴)
    - DoS (🔴)
    - Brute Force (🔴)
    - Botnet (🔴)
    - PortScan (🔴)
    - Infiltration (🔴)
    """

    ATTACK_CLASSES = ['Normal', 'DDoS', 'DoS', 'Brute Force', 'Botnet', 'PortScan', 'Infiltration']
    PROTOCOLS = ['TCP', 'UDP', 'ICMP']

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def generate_dataset(self, num_samples: int = 5000) -> pd.DataFrame:
        records = []
        
        # Define class weights: 55% Normal, 45% Attacks spread across types
        weights = [0.55, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04]
        labels = np.random.choice(self.ATTACK_CLASSES, size=num_samples, p=weights)

        for i, label in enumerate(labels):
            src_ip = f"192.168.1.{np.random.randint(2, 254)}"
            dst_ip = f"10.0.0.{np.random.randint(2, 50)}"
            
            if label == 'Normal':
                protocol = np.random.choice(self.PROTOCOLS, p=[0.75, 0.20, 0.05])
                src_port = np.random.randint(1024, 65535)
                dst_port = np.random.choice([80, 443, 53, 22, 8080])
                duration_sec = np.random.exponential(scale=2.5) + 0.1
                fwd_pkts = np.random.randint(2, 100)
                bwd_pkts = np.random.randint(2, 120)
                pkt_size_mean = np.random.normal(loc=512, scale=120)
                pkt_size_std = np.random.uniform(20, 150)
                pkt_size_min = max(40, pkt_size_mean - 2 * pkt_size_std)
                pkt_size_max = pkt_size_mean + 2 * pkt_size_std
                syn_flag = np.random.choice([0, 1], p=[0.8, 0.2])
                ack_flag = np.random.choice([0, 1], p=[0.1, 0.9])
                rst_flag = 0
                fin_flag = np.random.choice([0, 1], p=[0.7, 0.3])
                psh_flag = np.random.choice([0, 1], p=[0.5, 0.5])
                is_attack = 0

            elif label == 'DDoS':
                protocol = np.random.choice(['TCP', 'UDP'], p=[0.7, 0.3])
                src_port = np.random.randint(1024, 65535)
                dst_port = np.random.choice([80, 443])
                duration_sec = np.random.exponential(scale=0.2) + 0.01
                fwd_pkts = np.random.randint(500, 5000)
                bwd_pkts = np.random.randint(0, 10)
                pkt_size_mean = np.random.normal(loc=64, scale=10)
                pkt_size_std = np.random.uniform(1, 10)
                pkt_size_min = 40
                pkt_size_max = 128
                syn_flag = 1
                ack_flag = 0
                rst_flag = np.random.choice([0, 1], p=[0.9, 0.1])
                fin_flag = 0
                psh_flag = 0
                is_attack = 1

            elif label == 'DoS':
                protocol = 'TCP'
                src_port = np.random.randint(1024, 65535)
                dst_port = 80
                duration_sec = np.random.exponential(scale=5.0) + 1.0
                fwd_pkts = np.random.randint(100, 1500)
                bwd_pkts = np.random.randint(0, 5)
                pkt_size_mean = np.random.normal(loc=120, scale=30)
                pkt_size_std = np.random.uniform(5, 25)
                pkt_size_min = 54
                pkt_size_max = 200
                syn_flag = 1
                ack_flag = 0
                rst_flag = 0
                fin_flag = 0
                psh_flag = 0
                is_attack = 1

            elif label == 'Brute Force':
                protocol = 'TCP'
                src_port = np.random.randint(1024, 65535)
                dst_port = np.random.choice([22, 21, 3389])  # SSH, FTP, RDP
                duration_sec = np.random.uniform(0.5, 3.0)
                fwd_pkts = np.random.randint(20, 80)
                bwd_pkts = np.random.randint(15, 75)
                pkt_size_mean = np.random.normal(loc=350, scale=50)
                pkt_size_std = np.random.uniform(10, 40)
                pkt_size_min = 60
                pkt_size_max = 600
                syn_flag = 1
                ack_flag = 1
                rst_flag = np.random.choice([0, 1], p=[0.6, 0.4])
                fin_flag = np.random.choice([0, 1], p=[0.5, 0.5])
                psh_flag = 1
                is_attack = 1

            elif label == 'Botnet':
                protocol = 'TCP'
                src_port = np.random.randint(1024, 65535)
                dst_port = np.random.choice([6667, 8080, 443])
                duration_sec = np.random.uniform(10.0, 120.0)
                fwd_pkts = np.random.randint(5, 30)
                bwd_pkts = np.random.randint(5, 30)
                pkt_size_mean = np.random.normal(loc=200, scale=30)
                pkt_size_std = np.random.uniform(5, 20)
                pkt_size_min = 50
                pkt_size_max = 400
                syn_flag = 1
                ack_flag = 1
                rst_flag = 0
                fin_flag = 0
                psh_flag = np.random.choice([0, 1], p=[0.8, 0.2])
                is_attack = 1

            elif label == 'PortScan':
                protocol = 'TCP'
                src_port = np.random.randint(1024, 65535)
                dst_port = np.random.randint(1, 1024)
                duration_sec = np.random.uniform(0.001, 0.05)
                fwd_pkts = np.random.randint(1, 3)
                bwd_pkts = np.random.randint(0, 2)
                pkt_size_mean = np.random.normal(loc=44, scale=4)
                pkt_size_std = np.random.uniform(0, 5)
                pkt_size_min = 40
                pkt_size_max = 60
                syn_flag = 1
                ack_flag = 0
                rst_flag = np.random.choice([0, 1], p=[0.3, 0.7])
                fin_flag = 0
                psh_flag = 0
                is_attack = 1

            elif label == 'Infiltration':
                protocol = 'TCP'
                src_port = np.random.randint(1024, 65535)
                dst_port = np.random.choice([4444, 5555, 80])
                duration_sec = np.random.uniform(5.0, 60.0)
                fwd_pkts = np.random.randint(50, 300)
                bwd_pkts = np.random.randint(40, 250)
                pkt_size_mean = np.random.normal(loc=700, scale=150)
                pkt_size_std = np.random.uniform(50, 200)
                pkt_size_min = 64
                pkt_size_max = 1460
                syn_flag = 1
                ack_flag = 1
                rst_flag = 0
                fin_flag = 1
                psh_flag = 1
                is_attack = 1

            total_pkts = fwd_pkts + bwd_pkts
            total_bytes = total_pkts * pkt_size_mean
            flow_bytes_sec = total_bytes / max(duration_sec, 1e-4)
            flow_pkts_sec = total_pkts / max(duration_sec, 1e-4)

            records.append({
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': int(src_port),
                'dst_port': int(dst_port),
                'protocol': protocol,
                'duration_sec': round(float(duration_sec), 4),
                'fwd_packets': int(fwd_pkts),
                'bwd_packets': int(bwd_pkts),
                'total_packets': int(total_pkts),
                'packet_size_mean': round(float(max(40, pkt_size_mean)), 2),
                'packet_size_min': round(float(max(40, pkt_size_min)), 2),
                'packet_size_max': round(float(max(pkt_size_mean, pkt_size_max)), 2),
                'packet_size_std': round(float(max(0, pkt_size_std)), 2),
                'flow_bytes_per_sec': round(float(flow_bytes_sec), 2),
                'flow_packets_per_sec': round(float(flow_pkts_sec), 2),
                'syn_flag': int(syn_flag),
                'ack_flag': int(ack_flag),
                'rst_flag': int(rst_flag),
                'fin_flag': int(fin_flag),
                'psh_flag': int(psh_flag),
                'label': label,
                'is_attack': int(is_attack)
            })

        return pd.DataFrame(records)

if __name__ == "__main__":
    generator = NetworkDatasetGenerator()
    df = generator.generate_dataset(100)
    print("Sample generated dataset shape:", df.shape)
    print(df.head())
