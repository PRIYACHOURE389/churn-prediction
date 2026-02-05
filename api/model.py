"""
Business-friendly churn scoring logic
(temporary rule-based model, ML-ready)
"""

MODEL_VERSION = "v1.0-business"


def predict_churn(customer: dict) -> dict:
    """
    Predict churn risk in a way NON-TECHNICAL users can understand.
    """

    score = 0.0

    # --- Business rules (simple & explainable) ---
    if customer.get("Contract") == "Month-to-month":
        score += 0.30

    if customer.get("InternetService") == "Fiber optic":
        score += 0.20

    if customer.get("PaymentMethod") == "Electronic check":
        score += 0.20

    if customer.get("tenure", 0) < 12:
        score += 0.20

    if customer.get("SeniorCitizen", 0) == 1:
        score += 0.10

    # Clamp score
    churn_probability = round(min(score, 0.95), 2)

    # --- Human interpretation ---
    if churn_probability >= 0.7:
        risk = "High"
        message = "Customer is very likely to leave"
        action = "Offer discount or long-term contract immediately"

    elif churn_probability >= 0.4:
        risk = "Medium"
        message = "Customer may leave in near future"
        action = "Engage with loyalty offers or follow-up"

    else:
        risk = "Low"
        message = "Customer likely to stay"
        action = "No immediate action required"

    # --- FINAL RESPONSE ---
    return {
        "churn_probability": churn_probability,
        "risk_level": risk,
        "explanation": message,
        "recommended_action": action,
        "model_version": MODEL_VERSION,
    }
