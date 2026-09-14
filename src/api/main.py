import os
import sys
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pipeline.predictor import NetworkAttackPredictor
from src.data.dataset_generator import NetworkDatasetGenerator

app = FastAPI(
    title="🔐 Cybersecurity Network Attack Detection API",
    description="Production-grade REST API for real-time network traffic threat detection, attack categorization, and anomaly scoring.",
    version="1.0.0"
)

# Global Predictor instance
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        models_dir = os.path.join('models', 'saved')
        predictor = NetworkAttackPredictor(models_dir=models_dir)
    return predictor

class TrafficPacketSchema(BaseModel):
    src_ip: str = Field("192.168.1.50", example="192.168.1.50")
    dst_ip: str = Field("10.0.0.5", example="10.0.0.5")
    src_port: int = Field(45210, example=45210)
    dst_port: int = Field(80, example=80)
    protocol: str = Field("TCP", example="TCP")
    duration_sec: float = Field(0.01, example=0.01)
    fwd_packets: int = Field(2000, example=2000)
    bwd_packets: int = Field(2, example=2)
    total_packets: int = Field(2002, example=2002)
    packet_size_mean: float = Field(64.0, example=64.0)
    packet_size_min: float = Field(40.0, example=40.0)
    packet_size_max: float = Field(128.0, example=128.0)
    packet_size_std: float = Field(5.0, example=5.0)
    flow_bytes_per_sec: float = Field(1281280.0, example=1281280.0)
    flow_packets_per_sec: float = Field(200200.0, example=200200.0)
    syn_flag: int = Field(1, example=1)
    ack_flag: int = Field(0, example=0)
    rst_flag: int = Field(0, example=0)
    fin_flag: int = Field(0, example=0)
    psh_flag: int = Field(0, example=0)

class BatchPacketsSchema(BaseModel):
    packets: List[TrafficPacketSchema]

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "Cybersecurity Network Attack Detection API",
        "model_loaded": get_predictor() is not None
    }

@app.post("/predict")
def predict_packet(packet: TrafficPacketSchema, model_type: str = "Random Forest"):
    try:
        pred_engine = get_predictor()
        res = pred_engine.predict_single(packet.dict(), model_type=model_type)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch_packets(data: BatchPacketsSchema, model_type: str = "Random Forest"):
    try:
        pred_engine = get_predictor()
        packets_dicts = [p.dict() for p in data.packets]
        import pandas as pd
        df = pd.DataFrame(packets_dicts)
        res = pred_engine.predict_batch(df, model_type=model_type)
        return {"total": len(res), "predictions": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulate")
def simulate_live_packet():
    gen = NetworkDatasetGenerator()
    df_sample = gen.generate_dataset(1)
    sample_dict = df_sample.iloc[0].to_dict()
    pred_engine = get_predictor()
    res = pred_engine.predict_single(sample_dict)
    return {
        "raw_packet": sample_dict,
        "prediction": res
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
