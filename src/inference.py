import warnings
from pathlib import Path
import joblib
import pandas as pd

# -------------------------------------------------------------------
# Silence safe encoder warnings
# -------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="Found unknown categories in columns"
)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
SCHEMA_PATH = PROJECT_ROOT / "models" / "raw_feature_schema.joblib"


class ChurnPredictor:
    """
    Single-source inference logic.
    Reusable across CLI, API, batch, and UI.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.pipeline = joblib.load(PIPELINE_PATH)
        self.expected_features = set(joblib.load(SCHEMA_PATH))

    def _validate_schema(self, df: pd.DataFrame):
        incoming = set(df.columns)
        missing = self.expected_features - incoming
        if missing:
            raise ValueError(f"Missing required features: {missing}")

    def predict(self, customer_data: dict) -> dict:
        df = pd.DataFrame([customer_data])
        self._validate_schema(df)

        probability = self.pipeline.predict_proba(df)[0, 1]
        prediction = int(probability >= self.threshold)

        return {
            "churn_probability": round(float(probability), 4),
            "churn_prediction": prediction,
            "threshold": self.threshold
        }
