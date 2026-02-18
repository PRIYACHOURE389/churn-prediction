from fastapi import APIRouter, HTTPException
from typing import List, Dict
import pandas as pd
import shap
from api.services.predictor import model, assign_risk


router = APIRouter(prefix="/predict", tags=["Prediction"])

# ----------------------------------------
# Batch Prediction
# ----------------------------------------
@router.post("/batch")
async def predict_batch(customers: List[Dict]):
    """
    Expects raw JSON list:
    [
        {...},
        {...}
    ]
    """

    try:
        # Convert list → DataFrame
        df = pd.DataFrame(customers)

        if df.empty:
            raise HTTPException(status_code=400, detail="Empty input data.")

        # Drop unwanted columns
        for col in ["customerID", "Churn"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Align with model training features
        if hasattr(model, "feature_names_in_"):
            df = df.reindex(columns=model.feature_names_in_)

        # Predict probabilities
        probs = model.predict_proba(df)[:, 1]

        results = [
            {
                "churn_probability": float(prob),
                "risk_segment": assign_risk(prob)
            }
            for prob in probs
        ]

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

