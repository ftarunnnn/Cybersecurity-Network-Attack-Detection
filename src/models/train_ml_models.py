import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.preprocessing.preprocessor import NetworkDataPreprocessor
from src.preprocessing.feature_engineer import NetworkFeatureEngineer
from src.models.ml_models import NetworkMLTrainer

def main():
    print("[Phase 6] Training Classical Machine Learning Models (Random Forest & XGBoost)...")
    
    train_path = os.path.join('data', 'raw', 'network_traffic.csv')
    test_path = os.path.join('data', 'raw', 'network_traffic_test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("[ERR] Training or test data not found. Run python data/generate_network_data.py first.")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 1. Feature Engineering
    fe = NetworkFeatureEngineer(top_k=15)
    train_df_fe = fe.create_features(train_df)
    test_df_fe = fe.create_features(test_df)

    # 2. Preprocessing
    preprocessor = NetworkDataPreprocessor()
    preprocessor.fit(train_df_fe)
    
    models_dir = os.path.join('models', 'saved')
    preprocessor.save(os.path.join(models_dir, 'preprocessor.joblib'))

    train_data = preprocessor.transform(train_df_fe)
    test_data = preprocessor.transform(test_df_fe)

    X_train, y_train_bin, y_train_mul = train_data['X'], train_data['y_binary'], train_data['y_multi']
    X_test, y_test_bin, y_test_mul = test_data['X'], test_data['y_binary'], test_data['y_multi']

    # 3. Train ML Models
    trainer = NetworkMLTrainer(random_state=42)
    trainer.train_random_forest(X_train, y_train_bin, y_train_mul)
    trainer.train_xgboost(X_train, y_train_bin, y_train_mul)

    # 4. Evaluate
    rf_bin_metrics = trainer.evaluate_model(trainer.rf_binary, X_test, y_test_bin, is_binary=True)
    rf_mul_metrics = trainer.evaluate_model(trainer.rf_multi, X_test, y_test_mul, is_binary=False)
    xgb_bin_metrics = trainer.evaluate_model(trainer.xgb_binary, X_test, y_test_bin, is_binary=True)
    xgb_mul_metrics = trainer.evaluate_model(trainer.xgb_multi, X_test, y_test_mul, is_binary=False)

    print("\n--- Model Evaluation Results ---")
    print(f"Random Forest (Binary)     -> Acc: {rf_bin_metrics['accuracy']}, Recall: {rf_bin_metrics['recall']}, F1: {rf_bin_metrics['f1_score']}")
    print(f"Random Forest (Multi-class)-> Acc: {rf_mul_metrics['accuracy']}, Recall: {rf_mul_metrics['recall']}, F1: {rf_mul_metrics['f1_score']}")
    print(f"XGBoost (Binary)           -> Acc: {xgb_bin_metrics['accuracy']}, Recall: {xgb_bin_metrics['recall']}, F1: {xgb_bin_metrics['f1_score']}")
    print(f"XGBoost (Multi-class)      -> Acc: {xgb_mul_metrics['accuracy']}, Recall: {xgb_mul_metrics['recall']}, F1: {xgb_mul_metrics['f1_score']}")

    # 5. Save Models
    trainer.save_models(models_dir)

if __name__ == '__main__':
    main()
