import pandas as pd
import logging
from pathlib import Path
from inference import predict_churn

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_clean.csv"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "predictions.csv"

TARGET_COL = "Churn"

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    logging.info("Starting batch prediction")

    df = pd.read_csv(INPUT_DATA_PATH)

    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    preds = predict_churn(df, threshold=0.4)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(OUTPUT_PATH, index=False)

    logging.info(f"Predictions saved to: {OUTPUT_PATH}")
    logging.info("Batch prediction completed successfully")


if __name__ == "__main__":
    main()
