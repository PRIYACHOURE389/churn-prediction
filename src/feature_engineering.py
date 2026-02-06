import pandas as pd
import logging
import json
from pathlib import Path

# -------------------------
# Logging Configuration
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------
# Project Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_clean.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# -------------------------
# Data Contract
# -------------------------
TARGET_COL = "Churn"

# -------------------------
# Feature Engineering Utilities
# -------------------------
def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def identify_feature_types(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    return categorical_cols, numerical_cols


def generate_feature_metadata(X, categorical_cols, numerical_cols):
    """
    Generate feature metadata for documentation, monitoring, and governance.
    """
    metadata = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "categorical_features": categorical_cols,
        "numerical_features": numerical_cols,
        "feature_dtypes": X.dtypes.astype(str).to_dict(),
        "target": TARGET_COL
    }
    return metadata


# -------------------------
# Main Execution
# -------------------------
def main():
    logger.info("Starting feature engineering (metadata only)")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load processed data
    df = pd.read_csv(PROCESSED_DATA_PATH)
    logger.info(f"Processed data loaded: {df.shape}")

    X, _ = split_features_target(df)

    categorical_cols, numerical_cols = identify_feature_types(X)

    logger.info(f"Categorical features: {len(categorical_cols)}")
    logger.info(f"Numerical features: {len(numerical_cols)}")

    metadata = generate_feature_metadata(
        X,
        categorical_cols,
        numerical_cols
    )

    # Persist feature metadata (NOT transformed data)
    metadata_path = ARTIFACTS_DIR / "feature_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Feature metadata saved to: {metadata_path}")
    logger.info("Feature engineering completed successfully")


if __name__ == "__main__":
    main()
