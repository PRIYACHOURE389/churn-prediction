# api/services/model_service.py
from api.model import model, MODEL_VERSION

def risk_bucket(prob: float) -> str:
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    return "Low Risk"

def retention_action(risk: str) -> str:
    return {
        "High Risk": "Offer discount + priority support",
        "Medium Risk": "Proactive engagement & loyalty offers",
        "Low Risk": "Upsell & value-added services"
    }[risk]

def predict_single(customer):
    X = [customer.dict().values()]
    prob = model.predict_proba(X)[0][1]

    risk = risk_bucket(prob)

    return {
        "churn_probability": round(float(prob), 3),
        "risk_segment": risk,
        "retention_action": retention_action(risk),
        "explanation": f"Customer classified as {risk} based on usage, contract, and tenure patterns.",
        "model_version": MODEL_VERSION
    }

def predict_batch(customers):
    return {
        "results": [predict_single(c) for c in customers]
    }