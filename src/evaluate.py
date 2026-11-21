"""
Chandrakant (Member 1) - Metrics, K-Fold cross-validation, comparison table.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score


def regression_metrics(name, y_true, y_pred):
    """RMSE, MAE and R2 for one model."""
    result = {
        "model": name,
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }
    print(result)
    return result


def cross_validate(model, X_train, y_train, k=5):
    """K-Fold CV on the training set. Returns mean and std of RMSE."""
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kfold,
                             scoring="neg_root_mean_squared_error", n_jobs=-1)
    rmse_scores = -scores
    print(f"  {k}-Fold CV RMSE: {rmse_scores.mean():.3f} +/- {rmse_scores.std():.3f}")
    return round(float(rmse_scores.mean()), 3), round(float(rmse_scores.std()), 3)


def save_results(results, out_dir="results"):
    """Save all model scores to JSON + CSV and print a comparison table."""
    os.makedirs(out_dir, exist_ok=True)
    table = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)

    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(table.to_dict(orient="records"), f, indent=2)
    table.to_csv(f"{out_dir}/model_comparison.csv", index=False)

    print("\n===== MODEL COMPARISON (sorted by RMSE) =====")
    print(table.to_string(index=False))
    return table