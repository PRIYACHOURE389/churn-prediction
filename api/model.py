import pandas as pd
import joblib
from pathlib import Path

MODEL_VERSION = "v1.0"

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"

_pipeline = None


def load_model():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predict_churn(data: dict):
    model = load_model()

    df = pd.DataFrame([data])
    proba = model.predict_proba(df)[0][1]

    label = "High Risk" if proba >= 0.5 else "Low Risk"

    return {
        "churn_probability": round(float(proba), 3),
        "churn_prediction": label,
        "model_version": MODEL_VERSION,
    }
