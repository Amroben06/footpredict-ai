import os
import sys
import re
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

DATA_PATH = os.path.join("data", "processed", "clean_matches.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "match_predictor.joblib")


def clean_numeric(val):
    """Clean strings with footnotes (e.g. '32[d]') and return float/int."""
    if pd.isna(val):
        return 0.0
    clean_str = re.sub(r"[^\d\-+.]", "", str(val))
    try:
        return float(clean_str)
    except ValueError:
        return 0.0


def train_model():
    logging.info("Starting model training pipeline...")

    if not os.path.exists(DATA_PATH):
        logging.error(f"Dataset not found at {DATA_PATH}. Run pipeline/clean_and_aggregate.py first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    # Sanitize numeric columns to prevent string arithmetic errors
    numeric_cols = ["home_points", "away_points", "home_goal_diff", "away_goal_diff", "home_goals", "away_goals"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    # Features: points and goal difference differential
    df["diff_points"] = df["home_points"] - df["away_points"]
    df["diff_goal_diff"] = df["home_goal_diff"] - df["away_goal_diff"]

    feature_cols = ["home_points", "away_points", "home_goal_diff", "away_goal_diff", "diff_points", "diff_goal_diff"]
    target_col = "outcome"

    X = df[feature_cols]
    y = df[target_col]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Model trained successfully. Test Accuracy: {acc:.2f}")
    logging.info("\n" + classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    logging.info(f"Model artifact saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()