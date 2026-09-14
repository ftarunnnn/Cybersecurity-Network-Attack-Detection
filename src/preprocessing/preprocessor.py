import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class NetworkDataPreprocessor:
    """
    Handles Phase 3 Data Preprocessing for Network Attack Detection:
    - Deduplication
    - Missing & Infinity handling
    - Categorical feature encoding (Protocol, Labels)
    - Irrelevant feature removal
    - Numerical standardization (StandardScaler)
    """

    NUMERICAL_FEATURES = [
        'src_port', 'dst_port', 'duration_sec', 'fwd_packets', 'bwd_packets',
        'total_packets', 'packet_size_mean', 'packet_size_min', 'packet_size_max',
        'packet_size_std', 'flow_bytes_per_sec', 'flow_packets_per_sec',
        'syn_flag', 'ack_flag', 'rst_flag', 'fin_flag', 'psh_flag'
    ]
    
    CATEGORICAL_FEATURES = ['protocol']
    TARGET_LABEL_COL = 'label'
    TARGET_BINARY_COL = 'is_attack'

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.fitted = False
        self.classes_ = []

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Deduplicate
        initial_len = len(df)
        df = df.drop_duplicates()
        dedup_count = initial_len - len(df)
        
        # 2. Replace Inf & -Inf with NaN then fill with median or 0
        df = df.replace([np.inf, -np.inf], np.nan)
        
        for col in self.NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 0.0)
                
        return df

    def fit(self, df: pd.DataFrame):
        df_clean = self.clean_data(df)
        
        # Protocol One-Hot / Label encoding map
        if self.TARGET_LABEL_COL in df_clean.columns:
            self.label_encoder.fit(df_clean[self.TARGET_LABEL_COL])
            self.classes_ = list(self.label_encoder.classes_)
            
        # Fit scaler on numerical features
        num_data = self._transform_protocol(df_clean)[self.NUMERICAL_FEATURES + ['protocol_code']]
        self.scaler.fit(num_data)
        self.fitted = True
        return self

    def _transform_protocol(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        protocol_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2}
        if 'protocol' in df.columns:
            df['protocol_code'] = df['protocol'].map(protocol_map).fillna(0).astype(int)
        else:
            df['protocol_code'] = 0
        return df

    def transform(self, df: pd.DataFrame, is_training: bool = False):
        df_clean = self.clean_data(df)
        df_trans = self._transform_protocol(df_clean)
        
        feature_cols = self.NUMERICAL_FEATURES + ['protocol_code']
        X_num = df_trans[feature_cols]
        
        X_scaled = self.scaler.transform(X_num)
        
        y_binary = df_clean[self.TARGET_BINARY_COL].values if self.TARGET_BINARY_COL in df_clean.columns else None
        
        y_multi = None
        if self.TARGET_LABEL_COL in df_clean.columns:
            y_multi = self.label_encoder.transform(df_clean[self.TARGET_LABEL_COL])
            
        return {
            'X': X_scaled,
            'feature_names': feature_cols,
            'y_binary': y_binary,
            'y_multi': y_multi,
            'raw_df': df_clean
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[OK] Preprocessor saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)

if __name__ == '__main__':
    from src.data.dataset_generator import NetworkDatasetGenerator
    gen = NetworkDatasetGenerator()
    df = gen.generate_dataset(100)
    prep = NetworkDataPreprocessor()
    prep.fit(df)
    res = prep.transform(df)
    print("Preprocessed X shape:", res['X'].shape)
    print("Classes:", prep.classes_)
