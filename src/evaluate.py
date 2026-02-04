import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TARGET_COL = "Churn"
TOP_K = 0.2  # Top 20% customers for business KPIs

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------------------------
# Business KPI helpers
# -------------------------------------------------------------------
def precision_at_k(y_true, y_prob, k):
    cutoff = int(len(y_prob) * k)
    idx = np.argsort(y_prob)[::-1][:cutoff]
    return y_true.iloc[idx].mean()


def churn_capture_rate(y_true, y_prob, k):
    cutoff = int(len(y_prob) * k)
    idx = np.argsort(y_prob)[::-1][:cutoff]
    return y_true.iloc[idx].sum() / y_true.sum()


def lift_table(y_true, y_prob, n_bins=10):
    df = pd.DataFrame({"y": y_true, "prob": y_prob})
    df["decile"] = pd.qcut(df["prob"], q=n_bins, labels=False, duplicates="drop")

    lift = (
        df.groupby("decile")
        .agg(
            customers=("y", "count"),
            churners=("y", "sum"),
            churn_rate=("y", "mean")
        )
        .sort_index(ascending=False)
        .reset_index()
    )

    baseline = y_true.mean()
    lift["lift"] = lift["churn_rate"] / baseline
    return lift


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def main():
    logging.info("Starting model evaluation")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data & model
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    pipeline = joblib.load(MODEL_PATH)

    # Predictions
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = pipeline.predict(X)

    # -------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------
    roc_auc = roc_auc_score(y, y_prob)
    clf_report = classification_report(y, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred)

    # Business KPIs
    p_at_k = precision_at_k(y, y_prob, TOP_K)
    capture = churn_capture_rate(y, y_prob, TOP_K)
    lift_df = lift_table(y, y_prob)

    logging.info(f"ROC-AUC: {roc_auc:.4f}")
    logging.info(f"Precision@{int(TOP_K*100)}%: {p_at_k:.4f}")
    logging.info(f"Churn Capture@{int(TOP_K*100)}%: {capture:.4f}")

    # -------------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------------
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

    pd.DataFrame(clf_report).T.to_csv(
        ARTIFACTS_DIR / "classification_report.csv"
    )

    pd.DataFrame(
        conf_matrix,
        index=["Actual_NoChurn", "Actual_Churn"],
        columns=["Pred_NoChurn", "Pred_Churn"]
    ).to_csv(
        ARTIFACTS_DIR / "confusion_matrix.csv"
    )

    lift_df.to_csv(
        ARTIFACTS_DIR / "decile_lift.csv",
        index=False
    )

    with open(ARTIFACTS_DIR / "roc_auc.txt", "w") as f:
        f.write(f"ROC-AUC: {roc_auc:.4f}")

    logging.info("Evaluation artifacts saved successfully")
    logging.info(f"Artifacts path: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
