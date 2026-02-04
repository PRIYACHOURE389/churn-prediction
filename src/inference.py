import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Union

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------------------------
# Load pipeline (lazy-loaded)
# -------------------------------------------------------------------
_pipeline = None


def load_pipeline():
    global _pipeline
    if _pipeline is None:
        logging.info("Loading churn pipeline")
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


# -------------------------------------------------------------------
# Core inference function (SINGLE SOURCE)
# -------------------------------------------------------------------
def predict_churn(
    data: Union[pd.DataFrame, dict],
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Predict churn probabilities and labels from raw input data.

    Parameters
    ----------
    data : pd.DataFrame or dict
        Raw customer data (no target column).
    threshold : float
        Decision threshold for churn classification.

    Returns
    -------
    pd.DataFrame
        churn_probability, churn_prediction
    """

    pipeline = load_pipeline()

    if isinstance(data, dict):
        data = pd.DataFrame([data])

    logging.info(f"Running inference on {len(data)} samples")

    churn_proba = pipeline.predict_proba(data)[:, 1]
    churn_pred = (churn_proba >= threshold).astype(int)

    results = pd.DataFrame({
        "churn_probability": churn_proba,
        "churn_prediction": churn_pred
    })

    return results

if __name__ == "__main__":
    import pandas as pd

    sample = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "churn_clean.csv"
    ).drop(columns=["Churn"]).head(5)

    preds = predict_churn(sample)
    print(preds)
