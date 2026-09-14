import os
import torch
import joblib
import numpy as np
import pandas as pd

from src.preprocessing.preprocessor import NetworkDataPreprocessor
from src.preprocessing.feature_engineer import NetworkFeatureEngineer
from src.models.ml_models import NetworkMLTrainer
from src.models.dl_models import DLTrainerManager

class NetworkAttackPredictor:
    """
    Phase 9: Complete Unified Network Attack Prediction Engine.
    
    Flow:
    Network Traffic Input -> Feature Extraction -> Preprocessing -> ML & DL Ensembled Models -> Threat Prediction
    
    Output Format:
    {
      "status": "🟢 Normal" or "🔴 Attack",
      "is_attack": True/False,
      "attack_type": "DDoS" / "DoS" / "Brute Force" / "Botnet" / "PortScan" / "Infiltration" / "Normal",
      "confidence_percent": 96.5,
      "anomaly_score": 0.042,
      "is_zero_day_anomaly": False,
      "details": { ... }
    }
    """

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join('models', 'saved')
            
        self.models_dir = models_dir
        self.preprocessor = NetworkDataPreprocessor.load(os.path.join(models_dir, 'preprocessor.joblib'))
        self.feature_engineer = NetworkFeatureEngineer(top_k=15)
        
        # Load Classical ML Models
        self.ml_trainer = NetworkMLTrainer.load_models(models_dir)
        
        # Load DL Models
        input_dim = len(self.preprocessor.NUMERICAL_FEATURES) + 1
        num_classes = len(self.preprocessor.classes_)
        self.dl_manager = DLTrainerManager(input_dim=input_dim, num_classes=num_classes)
        self.dl_manager.load_all(models_dir)

    def predict_single(self, traffic_sample: dict, model_type: str = 'Random Forest') -> dict:
        df_raw = pd.DataFrame([traffic_sample])
        res = self.predict_batch(df_raw, model_type=model_type)
        return res[0]

    def predict_batch(self, df_raw: pd.DataFrame, model_type: str = 'Random Forest') -> list:
        # 1. Feature Engineering
        df_fe = self.feature_engineer.create_features(df_raw)
        
        # 2. Preprocessing
        trans_res = self.preprocessor.transform(df_fe)
        X_scaled = trans_res['X']
        
        results = []
        
        # 3. Model Inference
        if model_type == 'Random Forest' and self.ml_trainer.rf_multi:
            preds = self.ml_trainer.rf_multi.predict(X_scaled)
            probs = self.ml_trainer.rf_multi.predict_proba(X_scaled)
        elif model_type == 'XGBoost' and self.ml_trainer.xgb_multi:
            preds = self.ml_trainer.xgb_multi.predict(X_scaled)
            probs = self.ml_trainer.xgb_multi.predict_proba(X_scaled)
        elif model_type == '1D CNN (PyTorch)':
            preds, probs = self.dl_manager.predict_with_confidence(self.dl_manager.cnn_model, X_scaled)
            probs_matrix = np.zeros((len(preds), len(self.preprocessor.classes_)))
            for idx, (p, c) in enumerate(zip(preds, probs)):
                probs_matrix[idx, p] = c
            probs = probs_matrix
        elif model_type == 'LSTM (PyTorch)':
            preds, probs = self.dl_manager.predict_with_confidence(self.dl_manager.lstm_model, X_scaled)
            probs_matrix = np.zeros((len(preds), len(self.preprocessor.classes_)))
            for idx, (p, c) in enumerate(zip(preds, probs)):
                probs_matrix[idx, p] = c
            probs = probs_matrix
        else:
            preds = self.ml_trainer.rf_multi.predict(X_scaled)
            probs = self.ml_trainer.rf_multi.predict_proba(X_scaled)

        # 4. Autoencoder Anomaly Calculation
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        ae_errors = self.dl_manager.autoencoder.compute_reconstruction_error(X_t)

        for i in range(len(df_raw)):
            pred_idx = int(preds[i])
            attack_label = self.preprocessor.classes_[pred_idx]
            confidence = float(np.max(probs[i])) * 100.0
            is_attack = (attack_label != 'Normal')
            status = "ATTACK" if is_attack else "NORMAL"
            
            recon_error = float(ae_errors[i])
            is_zero_day = bool(recon_error > self.dl_manager.ae_threshold and not is_attack)

            results.append({
                'status': status,
                'is_attack': is_attack,
                'attack_type': attack_label,
                'confidence_percent': round(confidence, 2),
                'anomaly_score': round(recon_error, 4),
                'is_zero_day_anomaly': is_zero_day,
                'flow_info': {
                    'src_ip': str(df_raw.iloc[i].get('src_ip', 'N/A')),
                    'dst_ip': str(df_raw.iloc[i].get('dst_ip', 'N/A')),
                    'dst_port': int(df_raw.iloc[i].get('dst_port', 0)),
                    'protocol': str(df_raw.iloc[i].get('protocol', 'TCP'))
                }
            })

        return results

if __name__ == '__main__':
    predictor = NetworkAttackPredictor()
    sample = {
        'src_ip': '192.168.1.50',
        'dst_ip': '10.0.0.5',
        'src_port': 45210,
        'dst_port': 80,
        'protocol': 'TCP',
        'duration_sec': 0.01,
        'fwd_packets': 2000,
        'bwd_packets': 2,
        'total_packets': 2002,
        'packet_size_mean': 64.0,
        'packet_size_min': 40.0,
        'packet_size_max': 128.0,
        'packet_size_std': 5.0,
        'flow_bytes_per_sec': 1281280.0,
        'flow_packets_per_sec': 200200.0,
        'syn_flag': 1,
        'ack_flag': 0,
        'rst_flag': 0,
        'fin_flag': 0,
        'psh_flag': 0
    }
    result = predictor.predict_single(sample)
    print("Sample Inference Status:", result['status'])
    print("Attack Type:", result['attack_type'])
    print("Confidence:", result['confidence_percent'], "%")
    print("Anomaly Score:", result['anomaly_score'])
