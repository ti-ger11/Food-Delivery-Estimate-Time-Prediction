"""
Om - Batch predictions for Kaggle's test.csv (no target column).
Run from the project root:  python predict_test.py
Output: results/predictions.csv  (ID, predicted_time_min)
"""
import os

import pandas as pd

from src.predict import build_features, load_artifacts, predict

TEST_PATH = "data/raw/test.csv"
OUT_PATH = "results/predictions.csv"


def main():
    raw = pd.read_csv(TEST_PATH)
    print(f"Loaded {len(raw)} new orders from {TEST_PATH}")

    features = build_features(raw)            # clean + features, no rows dropped
    preds = predict(features)

    out = pd.DataFrame({
        "ID": raw["ID"].str.strip(),
        "predicted_time_min": preds.round(1),
    })
    os.makedirs("results", exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Model used: {load_artifacts()['model_name']}")
    print(f"Saved {len(out)} predictions to {OUT_PATH}")
    print(out["predicted_time_min"].describe().round(1))


if __name__ == "__main__":
    main()
