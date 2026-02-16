import joblib
from pathlib import Path

MODEL_PATH = Path("models/churn_pipeline.joblib")

model = joblib.load(MODEL_PATH)
