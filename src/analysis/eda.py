import os
import json
import numpy as np
import pandas as pd

class NetworkEDAAnalyzer:
    """
    Phase 4: Exploratory Data Analysis & Network Traffic Analyzer
    Computes statistical metrics, correlation matrix, outlier counts,
    and class imbalance summaries.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.num_cols = [
            'duration_sec', 'fwd_packets', 'bwd_packets', 'total_packets',
            'packet_size_mean', 'packet_size_min', 'packet_size_max', 'packet_size_std',
            'flow_bytes_per_sec', 'flow_packets_per_sec'
        ]

    def compute_summary(self) -> dict:
        total_records = len(self.df)
        
        # Binary distribution
        is_attack_counts = self.df['is_attack'].value_counts().to_dict() if 'is_attack' in self.df.columns else {}
        attack_ratio = float(is_attack_counts.get(1, 0) / max(1, total_records))
        
        # Multi-class distribution
        label_counts = self.df['label'].value_counts().to_dict() if 'label' in self.df.columns else {}
        
        # Outlier detection via IQR
        outliers = {}
        for col in self.num_cols:
            if col in self.df.columns:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count = int(((self.df[col] < lower) | (self.df[col] > upper)).sum())
                outliers[col] = {
                    'q1': float(q1),
                    'q3': float(q3),
                    'iqr': float(iqr),
                    'outlier_count': outlier_count,
                    'outlier_ratio': round(float(outlier_count / max(1, total_records)), 4)
                }

        # Mean comparison between Normal and Attack
        group_means = {}
        if 'is_attack' in self.df.columns:
            grouped = self.df.groupby('is_attack')[self.num_cols].mean()
            for col in self.num_cols:
                if col in grouped.columns:
                    group_means[col] = {
                        'Normal_mean': round(float(grouped.loc[0, col]), 2) if 0 in grouped.index else 0.0,
                        'Attack_mean': round(float(grouped.loc[1, col]), 2) if 1 in grouped.index else 0.0,
                    }

        # Feature correlation matrix
        corr_matrix = {}
        existing_num = [c for c in self.num_cols if c in self.df.columns]
        if existing_num:
            corr_df = self.df[existing_num].corr().round(3)
            corr_matrix = corr_df.to_dict()

        summary = {
            'total_records': total_records,
            'binary_distribution': is_attack_counts,
            'attack_percentage': round(attack_ratio * 100, 2),
            'label_distribution': label_counts,
            'group_means': group_means,
            'outliers_iqr': outliers,
            'correlation_matrix': corr_matrix
        }

        return summary

    def save_summary_json(self, filepath: str):
        summary = self.compute_summary()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] EDA Summary saved to: {filepath}")
        return summary

if __name__ == '__main__':
    train_path = os.path.join('data', 'raw', 'network_traffic.csv')
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        analyzer = NetworkEDAAnalyzer(df)
        summary = analyzer.save_summary_json(os.path.join('reports', 'eda_summary.json'))
        print("Total Records analyzed:", summary['total_records'])
        print("Attack Percentage:", summary['attack_percentage'], "%")
