import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
X_TEST_PATH = PROJECT_ROOT / "models" / "X_test.csv"
Y_TEST_PATH = PROJECT_ROOT / "models" / "y_test.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TOP_K = 0.2

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Business KPIs
# -------------------------------------------------------------------
def precision_at_k(y_true, y_prob, k):
    cutoff = int(len(y_prob) * k)
    idx = np.argsort(y_prob)[::-1][:cutoff]
    return y_true.iloc[idx].mean()


def churn_capture_rate(y_true, y_prob, k):
    cutoff = int(len(y_prob) * k)
    idx = np.argsort(y_prob)[::-1][:cutoff]
    return y_true.iloc[idx].sum() / y_true.sum()


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def main():
    logger.info("Starting model evaluation")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    p_at_k = precision_at_k(y_test, y_prob, TOP_K)
    capture = churn_capture_rate(y_test, y_prob, TOP_K)

    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"Precision@{int(TOP_K*100)}%: {p_at_k:.4f}")
    logger.info(f"Churn Capture@{int(TOP_K*100)}%: {capture:.4f}")

    with open(ARTIFACTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(
            {
                "roc_auc": roc_auc,
                "precision_at_k": p_at_k,
                "churn_capture_rate": capture,
                "top_k": TOP_K
            },
            f,
            indent=4
        )

    pd.DataFrame(report).T.to_csv(ARTIFACTS_DIR / "classification_report.csv")
    pd.DataFrame(
        cm,
        index=["Actual_NoChurn", "Actual_Churn"],
        columns=["Pred_NoChurn", "Pred_Churn"]
    ).to_csv(ARTIFACTS_DIR / "confusion_matrix.csv")

    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
