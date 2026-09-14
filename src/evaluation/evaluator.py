import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

class NetworkModelEvaluator:
    """
    Phase 8: Comprehensive Model Evaluation & Security Benchmark Suite.
    Calculates Accuracy, Precision, Recall (Security Priority), F1-Score,
    Confusion Matrix, and Multi-Class ROC-AUC.
    """

    def __init__(self, target_names: list = None):
        self.target_names = target_names or ['Normal', 'DDoS', 'DoS', 'Brute Force', 'Botnet', 'PortScan', 'Infiltration']

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
        acc = accuracy_score(y_true, y_pred)
        prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Per-class metrics
        prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate ROC-AUC if probability matrix is provided
        roc_auc = None
        if y_prob is not None:
            try:
                if y_prob.ndim == 2 and y_prob.shape[1] > 2:
                    roc_auc = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'))
                elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:
                    roc_auc = float(roc_auc_score(y_true, y_prob))
            except Exception as e:
                roc_auc = None

        class_breakdown = {}
        for idx, name in enumerate(self.target_names):
            if idx < len(prec_per_class):
                class_breakdown[name] = {
                    'precision': round(float(prec_per_class[idx]), 4),
                    'recall': round(float(rec_per_class[idx]), 4),
                    'f1_score': round(float(f1_per_class[idx]), 4)
                }

        return {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec_weighted), 4),
            'recall': round(float(rec_weighted), 4),
            'f1_score': round(float(f1_weighted), 4),
            'roc_auc': round(float(roc_auc), 4) if roc_auc is not None else 'N/A',
            'confusion_matrix': cm.tolist(),
            'class_breakdown': class_breakdown
        }

    def generate_evaluation_report(self, models_dict: dict, save_path: str = None) -> dict:
        """
        Takes a dict of {model_name: (y_true, y_pred, y_prob)} and computes comparison.
        """
        report = {}
        for model_name, (y_true, y_pred, y_prob) in models_dict.items():
            report[model_name] = self.evaluate(y_true, y_pred, y_prob)
            
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"[OK] Evaluation Metrics Report saved to: {save_path}")
            
        return report

if __name__ == '__main__':
    evaluator = NetworkModelEvaluator()
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1])
    res = evaluator.evaluate(y_true, y_pred)
    print("Sample Evaluation result:", res)
