import pandas as pd
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

REQUIRED_COLUMNS = {
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "InternetService",
    "MonthlyCharges",
    "TotalCharges",
    TARGET_COL
}


def load_data(path: Path) -> pd.DataFrame:
    logger.info(f"Loading data from: {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    initial_rows = len(df)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    missing_rows = df.isna().any(axis=1).sum()
    logger.info(f"Rows with missing values: {missing_rows}")

    df.dropna(inplace=True)

    logger.info(f"Dropped {initial_rows - len(df)} rows due to missing values")

    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    return df


def main():
    logger.info("Starting data preprocessing")

    df_raw = load_data(RAW_DATA_PATH)
    logger.info(f"Raw data shape: {df_raw.shape}")

    missing_cols = REQUIRED_COLUMNS - set(df_raw.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df_clean = clean_data(df_raw)
    logger.info(f"Cleaned data shape: {df_clean.shape}")

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)

    logger.info(f"Cleaned data saved to: {PROCESSED_DATA_PATH}")
    logger.info("Data preprocessing completed successfully")


if __name__ == "__main__":
    main()
