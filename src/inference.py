import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Union, Optional, List

# -------------------------------------------------------------------
# Paths & Version
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
MODEL_VERSION = "churn_pipeline_v1"

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logger = logging.getLogger("churn-inference")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# -------------------------------------------------------------------
# Lazy-loaded model
# -------------------------------------------------------------------
_pipeline = None


def load_pipeline():
    global _pipeline
    if _pipeline is None:
        logger.info(f"Loading model from {MODEL_PATH}")
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


# -------------------------------------------------------------------
# Business interpretation layer (IMPORTANT)
# -------------------------------------------------------------------
def interpret_probability(prob: float) -> dict:
    if prob >= 0.7:
        return {
            "risk_level": "High",
            "confidence": "Very likely to leave",
            "business_advice": "Offer discount or long-term contract immediately",
        }
    elif prob >= 0.4:
        return {
            "risk_level": "Medium",
            "confidence": "Might leave soon",
            "business_advice": "Engage with loyalty benefits or follow-up",
        }
    else:
        return {
            "risk_level": "Low",
            "confidence": "Likely to stay",
            "business_advice": "No immediate action required",
        }


# -------------------------------------------------------------------
# SINGLE prediction (used by /predict)
# -------------------------------------------------------------------
def predict_single(customer: dict) -> dict:
    model = load_pipeline()
    df = pd.DataFrame([customer])

    prob = float(model.predict_proba(df)[0][1])
    interpretation = interpret_probability(prob)

    return {
        "churn_probability": round(prob, 2),
        **interpretation,
        "model_version": MODEL_VERSION,
    }


# -------------------------------------------------------------------
# BATCH prediction (used by /batch-predict)
# -------------------------------------------------------------------
def predict_batch(customers: List[dict]) -> List[dict]:
    model = load_pipeline()
    df = pd.DataFrame(customers)

    probs = model.predict_proba(df)[:, 1]

    results = []
    for prob in probs:
        interpretation = interpret_probability(float(prob))
        results.append(
            {
                "churn_probability": round(float(prob), 2),
                **interpretation,
                "model_version": MODEL_VERSION,
            }
        )

    return results


# -------------------------------------------------------------------
# Local test (safe to keep)
# -------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        pd.read_csv(PROJECT_ROOT / "data" / "processed" / "churn_clean.csv")
        .drop(columns=["Churn"])
        .head(3)
        .to_dict(orient="records")
    )

    for r in predict_batch(sample):
        print(r)
