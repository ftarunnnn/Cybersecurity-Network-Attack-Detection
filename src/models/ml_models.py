import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

class NetworkMLTrainer:
    """
    Phase 6: Machine Learning Model Development for Network Traffic Classification.
    Trains:
    1. Random Forest (Binary & Multi-Class)
    2. XGBoost (Binary & Multi-Class)
    
    Optimized for high recall to prevent uncaught cyber attack breaches.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rf_binary = None
        self.rf_multi = None
        self.xgb_binary = None
        self.xgb_multi = None

    def train_random_forest(self, X_train: np.ndarray, y_train_binary: np.ndarray, y_train_multi: np.ndarray):
        print("[Training] Fitting Random Forest (Binary Classification)...")
        self.rf_binary = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.rf_binary.fit(X_train, y_train_binary)

        print("[Training] Fitting Random Forest (Multi-Class Attack Classification)...")
        self.rf_multi = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.rf_multi.fit(X_train, y_train_multi)

    def train_xgboost(self, X_train: np.ndarray, y_train_binary: np.ndarray, y_train_multi: np.ndarray):
        print("[Training] Fitting XGBoost (Binary Classification)...")
        self.xgb_binary = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=self.random_state
        )
        self.xgb_binary.fit(X_train, y_train_binary)

        print("[Training] Fitting XGBoost (Multi-Class Attack Classification)...")
        num_classes = len(np.unique(y_train_multi))
        self.xgb_multi = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss',
            random_state=self.random_state
        )
        self.xgb_multi.fit(X_train, y_train_multi)

    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, is_binary: bool = True) -> dict:
        y_pred = model.predict(X_test)
        
        if is_binary:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        else:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        return {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4)
        }

    def save_models(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        if self.rf_binary:
            joblib.dump(self.rf_binary, os.path.join(save_dir, 'rf_binary.joblib'))
        if self.rf_multi:
            joblib.dump(self.rf_multi, os.path.join(save_dir, 'rf_multi.joblib'))
        if self.xgb_binary:
            joblib.dump(self.xgb_binary, os.path.join(save_dir, 'xgb_binary.joblib'))
        if self.xgb_multi:
            joblib.dump(self.xgb_multi, os.path.join(save_dir, 'xgb_multi.joblib'))
        print(f"[OK] Classical ML models saved to: {save_dir}")

    @classmethod
    def load_models(cls, save_dir: str):
        trainer = cls()
        rf_bin_path = os.path.join(save_dir, 'rf_binary.joblib')
        rf_mul_path = os.path.join(save_dir, 'rf_multi.joblib')
        xgb_bin_path = os.path.join(save_dir, 'xgb_binary.joblib')
        xgb_mul_path = os.path.join(save_dir, 'xgb_multi.joblib')

        if os.path.exists(rf_bin_path):
            trainer.rf_binary = joblib.load(rf_bin_path)
        if os.path.exists(rf_mul_path):
            trainer.rf_multi = joblib.load(rf_mul_path)
        if os.path.exists(xgb_bin_path):
            trainer.xgb_binary = joblib.load(xgb_bin_path)
        if os.path.exists(xgb_mul_path):
            trainer.xgb_multi = joblib.load(xgb_mul_path)
            
        return trainer
