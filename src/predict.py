import logging
from src.inference import ChurnPredictor

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------------------------
# Sample payload (business features ONLY)
# Must match raw_feature_schema.joblib
# -------------------------------------------------------------------
SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.85,
    "TotalCharges": 456.45
}

# -------------------------------------------------------------------
# CLI Entrypoint
# -------------------------------------------------------------------
def main():
    logging.info("Running churn prediction")

    predictor = ChurnPredictor(threshold=0.5)

    result = predictor.predict(SAMPLE_CUSTOMER)

    print("\nPrediction Result")
    print("-" * 30)
    print(f"Churn Probability : {result['churn_probability']}")
    print(f"Churn Prediction  : {result['churn_prediction']}")
    print(f"Threshold         : {result['threshold']}")


if __name__ == "__main__":
    main()
