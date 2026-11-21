"""
Phase 2: featured_data.csv -> trained models -> metrics -> best model saved.
Run alone:  python run_phase2.py      (or through main.py)
"""
import json
import os

import joblib

from src.model_prep import prepare_data
from src.evaluate import regression_metrics, cross_validate, save_results
from src.train_ml import train_linear, train_random_forest, train_xgboost
from src.train_ann import tune_ann
from src.plots import actual_vs_predicted, residual_plot, feature_importance, ann_loss_curve


def main():
    os.makedirs("models", exist_ok=True)

    # Step 1: Prepare data (Chandrakant)
    X_train, X_test, y_train, y_test, feature_names = prepare_data()

    results = []
    models = {}
    predictions = {}

    # Step 2: Traditional ML models (Om) + CV (Chandrakant)
    print("\n--- Linear Regression ---")
    models["LinearRegression"] = train_linear(X_train, y_train)

    print("\n--- Random Forest ---")
    models["RandomForest"] = train_random_forest(X_train, y_train)

    print("\n--- XGBoost ---")
    models["XGBoost"] = train_xgboost(X_train, y_train)

    for name, model in models.items():
        print(f"\nEvaluating {name}")
        predictions[name] = model.predict(X_test)
        row = regression_metrics(name, y_test, predictions[name])
        row["cv_rmse_mean"], row["cv_rmse_std"] = cross_validate(model, X_train, y_train)
        results.append(row)

    # Step 3: ANN (Rshishabh)
    print("\n--- ANN (Keras) ---")
    ann, history = tune_ann(X_train, y_train)
    predictions["ANN"] = ann.predict(X_test.astype("float32"), verbose=0).ravel()
    row = regression_metrics("ANN", y_test, predictions["ANN"])
    row["cv_rmse_mean"], row["cv_rmse_std"] = None, None  # ANN uses a validation split instead
    results.append(row)
    ann.save("models/ann_model.keras")
    ann_loss_curve(history)

    # Step 4: Compare + pick best by lowest test RMSE (Chandrakant)
    table = save_results(results)
    best_name = table.iloc[0]["model"]
    print(f"\nBest model: {best_name}")

    if best_name == "ANN":
        best_path = "models/ann_model.keras"
    else:
        best_path = "models/best_model.pkl"
        joblib.dump(models[best_name], best_path)

    with open("results/best_model.json", "w") as f:
        json.dump({"best_model": best_name, "path": best_path}, f, indent=2)

    # Step 5: Charts for the best model (Rshishabh)
    actual_vs_predicted(y_test, predictions[best_name], best_name)
    residual_plot(y_test, predictions[best_name], best_name)
    tree_name = best_name if best_name in ("RandomForest", "XGBoost") else "XGBoost"
    feature_importance(models[tree_name], feature_names, tree_name)

    print("\nPhase 2 complete")


if __name__ == "__main__":
    main()