import pandas as pd

from api.schemas import CustomerFeatures
from api.model import model

MODEL_VERSION = "v1"

RECOMMENDATIONS = {
    "High": "Immediate retention action recommended",
    "Medium": "Targeted engagement advised",
    "Low": "No immediate action required"
}

def predict_single(customer: CustomerFeatures) -> dict:
    df = pd.DataFrame([customer.model_dump()])

    prob = model.predict_proba(df)[0][1]

    risk = (
        "High" if prob > 0.7 else
        "Medium" if prob > 0.4 else
        "Low"
    )

    return {
        "churn_probability": round(prob, 4),
        "risk_segment": risk,
        "recommendation": RECOMMENDATIONS[risk],
        "model_version": MODEL_VERSION
    }
