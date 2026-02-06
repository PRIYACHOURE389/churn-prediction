import logging
import json
import warnings
from pathlib import Path

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# -------------------------------------------------------------------
# Silence safe encoder warnings (production hygiene)
# -------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="Found unknown categories in columns"
)

# -------------------------------------------------------------------
# Paths & config
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_clean.csv"
MODEL_DIR = PROJECT_ROOT / "models"

PIPELINE_PATH = MODEL_DIR / "churn_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "training_metrics.json"
SCHEMA_PATH = MODEL_DIR / "raw_feature_schema.joblib"

TARGET_COL = "Churn"
ID_COLS = ["customerID"]          # 🔒 identifiers explicitly excluded
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into features and target.
    Identifiers are explicitly removed to guarantee
    a clean inference schema.
    """
    X = df.drop(columns=[TARGET_COL] + ID_COLS, errors="ignore")
    y = df[TARGET_COL]
    return X, y


def identify_feature_types(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_cols = X.select_dtypes(
        exclude=["object", "string"]
    ).columns.tolist()

    return categorical_cols, numerical_cols


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def main():
    logging.info("Starting model training")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X, y = split_features_target(df)

    # -------------------------------------------------------------------
    # Persist raw feature schema (INFERENCE CONTRACT)
    # -------------------------------------------------------------------
    joblib.dump(X.columns.tolist(), SCHEMA_PATH)

    categorical_cols, numerical_cols = identify_feature_types(X)

    logging.info(f"Samples: {X.shape[0]}")
    logging.info(f"Categorical features: {len(categorical_cols)}")
    logging.info(f"Numerical features: {len(numerical_cols)}")

    # -------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    min_frequency=0.01,
                    sparse_output=True
                ),
                categorical_cols
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_cols
            )
        ],
        remainder="drop"
    )

    # -------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------
    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    # -------------------------------------------------------------------
    # Train / validation split
    # -------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    pipeline.fit(X_train, y_train)

    # -------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    logging.info(f"Holdout ROC-AUC: {auc:.4f}")

    # -------------------------------------------------------------------
    # Persist artifacts
    # -------------------------------------------------------------------
    joblib.dump(pipeline, PIPELINE_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "roc_auc": auc,
                "classification_report": report,
                "n_samples": int(X.shape[0]),
                "n_raw_features": int(X.shape[1]),
                "excluded_columns": ID_COLS,
                "random_state": RANDOM_STATE
            },
            f,
            indent=4
        )

    logging.info("Model, schema, and metrics saved successfully")


if __name__ == "__main__":
    main()
