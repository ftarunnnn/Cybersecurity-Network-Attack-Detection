import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.preprocessing.preprocessor import NetworkDataPreprocessor
from src.preprocessing.feature_engineer import NetworkFeatureEngineer
from src.models.dl_models import DLTrainerManager

def main():
    print("[Phase 7] Training PyTorch Deep Learning Models (1D CNN, LSTM, Autoencoder)...")
    
    train_path = os.path.join('data', 'raw', 'network_traffic.csv')
    test_path = os.path.join('data', 'raw', 'network_traffic_test.csv')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("[ERR] Raw data not found. Run python data/generate_network_data.py first.")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    fe = NetworkFeatureEngineer(top_k=15)
    train_df_fe = fe.create_features(train_df)
    test_df_fe = fe.create_features(test_df)

    preprocessor = NetworkDataPreprocessor.load(os.path.join('models', 'saved', 'preprocessor.joblib'))
    train_data = preprocessor.transform(train_df_fe)
    test_data = preprocessor.transform(test_df_fe)

    X_train, y_train_mul = train_data['X'], train_data['y_multi']
    X_test, y_test_mul, y_test_bin = test_data['X'], test_data['y_multi'], test_data['y_binary']

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train_mul))

    dl_manager = DLTrainerManager(input_dim=input_dim, num_classes=num_classes)

    # 1. Train 1D CNN
    print("[Training] Fitting 1D CNN Classifier (15 epochs)...")
    dl_manager.train_classifier(dl_manager.cnn_model, X_train, y_train_mul, epochs=15, lr=0.001)

    # 2. Train LSTM
    print("[Training] Fitting LSTM Classifier (15 epochs)...")
    dl_manager.train_classifier(dl_manager.lstm_model, X_train, y_train_mul, epochs=15, lr=0.001)

    # 3. Train Autoencoder on Normal traffic only
    print("[Training] Fitting Autoencoder on Normal traffic (20 epochs)...")
    normal_mask = (y_train_mul == preprocessor.label_encoder.transform(['Normal'])[0])
    X_normal = X_train[normal_mask]
    dl_manager.train_autoencoder(X_normal, epochs=20, lr=0.001)

    # 4. Evaluate Models
    cnn_preds, cnn_conf = dl_manager.predict_with_confidence(dl_manager.cnn_model, X_test)
    lstm_preds, lstm_conf = dl_manager.predict_with_confidence(dl_manager.lstm_model, X_test)

    # Autoencoder evaluation
    import torch
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    ae_errors = dl_manager.autoencoder.compute_reconstruction_error(X_test_t)
    ae_preds_bin = (ae_errors > dl_manager.ae_threshold).astype(int)

    cnn_acc = accuracy_score(y_test_mul, cnn_preds)
    cnn_rec = recall_score(y_test_mul, cnn_preds, average='weighted', zero_division=0)
    
    lstm_acc = accuracy_score(y_test_mul, lstm_preds)
    lstm_rec = recall_score(y_test_mul, lstm_preds, average='weighted', zero_division=0)

    ae_acc = accuracy_score(y_test_bin, ae_preds_bin)
    ae_rec = recall_score(y_test_bin, ae_preds_bin, zero_division=0)

    print("\n--- Deep Learning Performance Results ---")
    print(f"1D CNN Classifier   -> Accuracy: {cnn_acc:.4f}, Recall: {cnn_rec:.4f}")
    print(f"LSTM Classifier     -> Accuracy: {lstm_acc:.4f}, Recall: {lstm_rec:.4f}")
    print(f"Autoencoder Anomaly -> Accuracy: {ae_acc:.4f}, Anomaly Recall: {ae_rec:.4f}")

    # 5. Save Model Artifacts
    models_dir = os.path.join('models', 'saved')
    dl_manager.save_all(models_dir)

if __name__ == '__main__':
    main()
