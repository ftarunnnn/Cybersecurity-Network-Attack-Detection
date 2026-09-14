import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class NetworkFeatureEngineer:
    """
    Phase 5: Feature Engineering & Selection Module for Network Traffic.
    Constructs high-value domain-specific cybersecurity indicators:
    - Packets per second
    - Bytes per second
    - Average packet size
    - SYN/ACK ratio
    - Payload byte ratio
    - Fwd/Bwd packet ratio
    - Protocol frequency
    
    Performs Feature Selection via Random Forest Gini Importance.
    """

    def __init__(self, top_k: int = 15):
        self.top_k = top_k
        self.selected_features = []
        self.feature_importances_ = {}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        duration = df['duration_sec'].clip(lower=1e-5)
        total_pkts = df['total_packets'].clip(lower=1)
        pkt_mean = df['packet_size_mean']
        pkt_max = df['packet_size_max'].clip(lower=1e-5)
        pkt_min = df['packet_size_min']
        fwd_pkts = df['fwd_packets']
        bwd_pkts = df['bwd_packets'].clip(lower=1e-5)
        syn = df['syn_flag']
        ack = df['ack_flag'].clip(lower=1e-5)

        # 1. Flow Rates
        df['derived_packets_per_sec'] = total_pkts / duration
        df['derived_bytes_per_sec'] = (total_pkts * pkt_mean) / duration
        
        # 2. Packet Geometry
        df['derived_avg_packet_size'] = (pkt_min + pkt_max + pkt_mean) / 3.0
        df['derived_payload_byte_ratio'] = pkt_mean / pkt_max
        
        # 3. Asymmetry & TCP Control Ratios
        df['derived_syn_ack_ratio'] = syn / ack
        df['derived_fwd_bwd_ratio'] = fwd_pkts / bwd_pkts
        
        # 4. Protocol Frequency Encoding
        if 'protocol' in df.columns:
            proto_counts = df['protocol'].value_counts(normalize=True).to_dict()
            df['derived_protocol_freq'] = df['protocol'].map(proto_counts).fillna(0.0)
        else:
            df['derived_protocol_freq'] = 0.5

        return df

    def fit_feature_selection(self, X: pd.DataFrame, y: np.ndarray, feature_names: list = None):
        if feature_names is None:
            feature_names = list(X.columns)

        selector_rf = RandomForestClassifier(n_estimators=50, random_state=42)
        selector_rf.fit(X, y)

        importances = selector_rf.feature_importances_
        self.feature_importances_ = {
            feature_names[i]: float(importances[i])
            for i in range(len(feature_names))
        }

        # Sort features by importance
        sorted_feats = sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [f[0] for f in sorted_feats[:self.top_k]]
        
        print(f"[OK] Selected top {len(self.selected_features)} features out of {len(feature_names)}")
        return self.selected_features

if __name__ == '__main__':
    from src.data.dataset_generator import NetworkDatasetGenerator
    from src.preprocessing.preprocessor import NetworkDataPreprocessor
    
    gen = NetworkDatasetGenerator()
    df = gen.generate_dataset(200)
    fe = NetworkFeatureEngineer(top_k=10)
    df_feat = fe.create_features(df)
    print("New derived features added:", [c for c in df_feat.columns if c.startswith('derived_')])
