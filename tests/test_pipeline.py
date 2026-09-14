import os
import sys
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_generator import NetworkDatasetGenerator
from src.preprocessing.preprocessor import NetworkDataPreprocessor
from src.preprocessing.feature_engineer import NetworkFeatureEngineer
from src.models.ml_models import NetworkMLTrainer
from src.models.dl_models import DLTrainerManager
from src.evaluation.evaluator import NetworkModelEvaluator
from src.pipeline.predictor import NetworkAttackPredictor
from src.api.main import app

@pytest.fixture
def sample_df():
    gen = NetworkDatasetGenerator(seed=42)
    return gen.generate_dataset(150)

def test_dataset_generator(sample_df):
    assert len(sample_df) == 150
    assert 'label' in sample_df.columns
    assert 'is_attack' in sample_df.columns
    assert 'duration_sec' in sample_df.columns

def test_preprocessor(sample_df):
    prep = NetworkDataPreprocessor()
    prep.fit(sample_df)
    res = prep.transform(sample_df)
    
    assert res['X'].shape[0] == len(sample_df)
    assert res['X'].shape[1] == len(prep.NUMERICAL_FEATURES) + 1
    assert len(prep.classes_) > 0

def test_feature_engineer(sample_df):
    fe = NetworkFeatureEngineer(top_k=10)
    df_feat = fe.create_features(sample_df)
    
    assert 'derived_packets_per_sec' in df_feat.columns
    assert 'derived_bytes_per_sec' in df_feat.columns
    assert 'derived_syn_ack_ratio' in df_feat.columns

def test_ml_models(sample_df):
    fe = NetworkFeatureEngineer(top_k=10)
    df_feat = fe.create_features(sample_df)
    
    prep = NetworkDataPreprocessor()
    prep.fit(df_feat)
    trans = prep.transform(df_feat)
    
    trainer = NetworkMLTrainer(random_state=42)
    trainer.train_random_forest(trans['X'], trans['y_binary'], trans['y_multi'])
    trainer.train_xgboost(trans['X'], trans['y_binary'], trans['y_multi'])
    
    rf_res = trainer.evaluate_model(trainer.rf_binary, trans['X'], trans['y_binary'], is_binary=True)
    assert rf_res['accuracy'] >= 0.8
    assert rf_res['recall'] >= 0.8

def test_dl_models(sample_df):
    fe = NetworkFeatureEngineer(top_k=10)
    df_feat = fe.create_features(sample_df)
    
    prep = NetworkDataPreprocessor()
    prep.fit(df_feat)
    trans = prep.transform(df_feat)
    
    input_dim = trans['X'].shape[1]
    num_classes = len(prep.classes_)
    
    dl_manager = DLTrainerManager(input_dim=input_dim, num_classes=num_classes)
    dl_manager.train_classifier(dl_manager.cnn_model, trans['X'], trans['y_multi'], epochs=2)
    dl_manager.train_classifier(dl_manager.lstm_model, trans['X'], trans['y_multi'], epochs=2)
    
    preds, confs = dl_manager.predict_with_confidence(dl_manager.cnn_model, trans['X'])
    assert len(preds) == len(sample_df)

def test_evaluator():
    evaluator = NetworkModelEvaluator(target_names=['Normal', 'Attack1', 'Attack2'])
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1])
    res = evaluator.evaluate(y_true, y_pred)
    
    assert 'accuracy' in res
    assert 'recall' in res
    assert 'confusion_matrix' in res

def test_predictor():
    models_dir = os.path.join('models', 'saved')
    if os.path.exists(os.path.join(models_dir, 'preprocessor.joblib')):
        predictor = NetworkAttackPredictor(models_dir=models_dir)
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
        res = predictor.predict_single(sample)
        assert 'status' in res
        assert 'attack_type' in res
        assert 'confidence_percent' in res

def test_fastapi_endpoints():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"
