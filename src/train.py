import pandas as pd
import logging
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "X_features.csv"
Y_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "retention_targets.csv"

X_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "X_test.csv"
Y_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "y_test.csv"

FEATURE_NAMES_PATH = PROJECT_ROOT / "models" / "feature_names.txt"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------------------------
# Helper functions (MUST be above main)
# -------------------------------------------------------------------
def load_training_data():
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    return X_train, y_train


def load_test_data():
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    return X_test, y_test


def load_feature_names():
    with open(FEATURE_NAMES_PATH) as f:
        return [line.strip() for line in f.readlines()]


def align_features(X, feature_names):
    """
    Enforce feature schema and order.
    """
    X = X.copy()

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names]
    return X


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def save_artifacts(model, scaler):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    logging.info("Starting model training")

    X_train, y_train = load_training_data()
    X_test, y_test = load_test_data()

    feature_names = load_feature_names()

    X_train = align_features(X_train, feature_names)
    X_test = align_features(X_test, feature_names)

    logging.info(f"Train shape: {X_train.shape}")
    logging.info(f"Test shape: {X_test.shape}")

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model = train_model(X_train_scaled, y_train)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    logging.info(f"Validation ROC-AUC: {roc_auc:.4f}")

    save_artifacts(model, scaler)

    logging.info("Training completed successfully")


if __name__ == "__main__":
    main()
