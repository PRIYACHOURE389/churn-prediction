import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

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

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "churn_raw.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_clean.csv"

# -------------------------
# Data Contract
# -------------------------
TARGET_COL = "Churn"


def load_data(path: Path) -> pd.DataFrame:
    """Load dataset from disk."""
    logger.info(f"Loading data from: {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset and enforce schema."""
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df.dropna(inplace=True)

    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    return df


def split_data(df, test_size=0.2, random_state=42):
    """Stratified train-test split."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def main():
    """
    Execute data preprocessing pipeline:
    raw data -> cleaned data -> saved to processed folder
    """
    logger.info("Starting data preprocessing")

    df_raw = load_data(RAW_DATA_PATH)
    logger.info(f"Raw data shape: {df_raw.shape}")

    df_clean = clean_data(df_raw)
    logger.info(f"Cleaned data shape: {df_clean.shape}")

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)

    logger.info(f"Cleaned data saved to: {PROCESSED_DATA_PATH}")
    logger.info("Data preprocessing completed successfully")


if __name__ == "__main__":
    main()
