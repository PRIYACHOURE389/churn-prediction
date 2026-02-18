import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"

model = joblib.load(MODEL_PATH)


def assign_risk(prob):
    if prob >= 0.75:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"
