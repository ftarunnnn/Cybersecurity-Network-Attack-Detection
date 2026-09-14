import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.preprocessing.preprocessor import NetworkDataPreprocessor
from src.preprocessing.feature_engineer import NetworkFeatureEngineer
from src.models.ml_models import NetworkMLTrainer
from src.models.dl_models import DLTrainerManager
from src.evaluation.evaluator import NetworkModelEvaluator

def main():
    print("[Phase 8] Running Comprehensive Model Evaluation Suite...")
    
    test_path = os.path.join('data', 'raw', 'network_traffic_test.csv')
    models_dir = os.path.join('models', 'saved')
    
    if not os.path.exists(test_path):
        print("[ERR] Test dataset not found.")
        return

    test_df = pd.read_csv(test_path)
    
    fe = NetworkFeatureEngineer(top_k=15)
    test_df_fe = fe.create_features(test_df)
    
    preprocessor = NetworkDataPreprocessor.load(os.path.join(models_dir, 'preprocessor.joblib'))
    test_data = preprocessor.transform(test_df_fe)
    
    X_test, y_test_mul, y_test_bin = test_data['X'], test_data['y_multi'], test_data['y_binary']

    evaluator = NetworkModelEvaluator(target_names=preprocessor.classes_)
    models_dict = {}

    # 1. Classical ML Models
    ml_trainer = NetworkMLTrainer.load_models(models_dir)
    if ml_trainer.rf_multi:
        rf_preds = ml_trainer.rf_multi.predict(X_test)
        rf_probs = ml_trainer.rf_multi.predict_proba(X_test)
        models_dict['Random Forest (Multi-Class)'] = (y_test_mul, rf_preds, rf_probs)
        
    if ml_trainer.xgb_multi:
        xgb_preds = ml_trainer.xgb_multi.predict(X_test)
        xgb_probs = ml_trainer.xgb_multi.predict_proba(X_test)
        models_dict['XGBoost (Multi-Class)'] = (y_test_mul, xgb_preds, xgb_probs)

    # 2. PyTorch Deep Learning Models
    input_dim = X_test.shape[1]
    num_classes = len(preprocessor.classes_)
    dl_manager = DLTrainerManager(input_dim=input_dim, num_classes=num_classes)
    dl_manager.load_all(models_dir)

    cnn_preds, cnn_conf = dl_manager.predict_with_confidence(dl_manager.cnn_model, X_test)
    models_dict['1D CNN (PyTorch)'] = (y_test_mul, cnn_preds, None)

    lstm_preds, lstm_conf = dl_manager.predict_with_confidence(dl_manager.lstm_model, X_test)
    models_dict['LSTM (PyTorch)'] = (y_test_mul, lstm_preds, None)

    report_path = os.path.join('reports', 'evaluation_metrics.json')
    report = evaluator.generate_evaluation_report(models_dict, save_path=report_path)

    print("\n--- Summary Evaluation Comparison ---")
    for model_name, metrics in report.items():
        print(f"[{model_name}] -> Accuracy: {metrics['accuracy']}, Recall: {metrics['recall']}, F1: {metrics['f1_score']}, ROC-AUC: {metrics['roc_auc']}")

if __name__ == '__main__':
    main()
