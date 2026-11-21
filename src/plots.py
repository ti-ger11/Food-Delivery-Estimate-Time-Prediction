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


def model_comparison(table, out_dir="images"):
    """Side-by-side bars of RMSE, MAE (lower is better) and R2 (higher is better)."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("rmse", "RMSE (min) - lower is better"),
               ("mae", "MAE (min) - lower is better"),
               ("r2", "R² - higher is better")]
    best = table.iloc[0]["model"]
    for ax, (col, title) in zip(axes, metrics):
        colors = ["tab:orange" if m == best else "tab:blue" for m in table["model"]]
        bars = ax.bar(table["model"], table[col], color=colors)
        ax.bar_label(bars, fmt="%.3f")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"Model comparison on the 20% test set (best: {best})")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/model_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"Model comparison chart saved to {out_dir}/model_comparison.png")


def all_models_actual_vs_predicted(y_true, predictions, out_dir="images"):
    """One small scatter per model so the models can be compared visually."""
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharex=True, sharey=True)
    lims = [y_true.min(), y_true.max()]
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ax.scatter(y_true, y_pred, alpha=0.2, s=6)
        ax.plot(lims, lims, "r--")
        ax.set_title(name)
        ax.set_xlabel("Actual (min)")
    axes[0].set_ylabel("Predicted (min)")
    fig.suptitle("Actual vs Predicted - all models")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/all_models_actual_vs_predicted.png", bbox_inches="tight")
    plt.close()