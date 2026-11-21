"""
Rshishabh - Evaluation charts (saved to images/).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def actual_vs_predicted(y_true, y_pred, name, out_dir="images"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual time (min)")
    plt.ylabel("Predicted time (min)")
    plt.title(f"Actual vs Predicted - {name}")
    plt.legend()
    plt.savefig(f"{out_dir}/actual_vs_predicted.png", bbox_inches="tight")
    plt.close()


def residual_plot(y_true, y_pred, name, out_dir="images"):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.3, s=10)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted time (min)")
    plt.ylabel("Residual (actual - predicted)")
    plt.title(f"Residual plot - {name}")
    plt.savefig(f"{out_dir}/residual_plot.png", bbox_inches="tight")
    plt.close()


def feature_importance(model, feature_names, name, out_dir="images", top=15):
    """Works for tree models (Random Forest / XGBoost)."""
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top)
    imp.index = [n.replace("num__", "").replace("cat__", "") for n in imp.index]
    plt.figure(figsize=(8, 6))
    imp[::-1].plot(kind="barh")
    plt.title(f"Top {top} features - {name}")
    plt.xlabel("Importance")
    plt.savefig(f"{out_dir}/feature_importance.png", bbox_inches="tight")
    plt.close()


def ann_loss_curve(history, out_dir="images"):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("ANN training curve")
    plt.legend()
    plt.savefig(f"{out_dir}/ann_loss_curve.png", bbox_inches="tight")
    plt.close()