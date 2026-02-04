import pandas as pd
import logging
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_clean.csv"

TARGET_COL = "Churn"

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------------------------
# Feature engineering functions
# -------------------------------------------------------------------
def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def identify_feature_types(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["string", "object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    return categorical_cols, numerical_cols


def encode_categoricals(X: pd.DataFrame, categorical_cols):
    """
    One-hot encode categorical features (for analysis only).
    NOTE: Do NOT persist encoded features to disk.
    """
    encoder = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    encoded_array = encoder.fit_transform(X[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X.index
    )

    return encoded_df


def build_feature_matrix(X, encoded_df, numerical_cols):
    """
    Combine numerical + encoded categorical features (in-memory only).
    """
    X_final = pd.concat(
        [X[numerical_cols], encoded_df],
        axis=1
    )
    return X_final


# -------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------
def main():
    logging.info("Starting feature engineering (NO persistence)")

    # Load cleaned data
    df = pd.read_csv(PROCESSED_DATA_PATH)
    logging.info(f"Loaded cleaned data: {df.shape}")

    # Split features and target
    X, y = split_features_target(df)

    # Identify feature types
    categorical_cols, numerical_cols = identify_feature_types(X)
    logging.info(f"Categorical features: {len(categorical_cols)}")
    logging.info(f"Numerical features: {len(numerical_cols)}")

    # Encode categoricals and build feature matrix (in-memory only)
    encoded_df = encode_categoricals(X, categorical_cols)
    X_final = build_feature_matrix(X, encoded_df, numerical_cols)

    logging.info(f"Final feature matrix shape (in-memory): {X_final.shape}")
    logging.info("Feature engineering completed (nothing saved to disk)")


if __name__ == "__main__":
    main()
