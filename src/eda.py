"""
Member 1 - EDA charts (saved to images/).
"""
import os

import matplotlib
matplotlib.use("Agg")  # save files without opening windows
import matplotlib.pyplot as plt
import seaborn as sns


def make_plots(df, out_dir="images"):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Distribution of the target
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Time_taken_min"], bins=30, kde=True)
    plt.title("Delivery time distribution")
    plt.xlabel("Minutes")
    plt.savefig(f"{out_dir}/time_distribution.png", bbox_inches="tight")
    plt.close()

    # 2. Correlation heatmap of number columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation heatmap")
    plt.savefig(f"{out_dir}/correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    # 3. Traffic vs delivery time
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Road_traffic_density", y="Time_taken_min", data=df,
                order=["Low", "Medium", "High", "Jam"])
    plt.title("Traffic density vs delivery time")
    plt.savefig(f"{out_dir}/traffic_vs_time.png", bbox_inches="tight")
    plt.close()

    print(f"Charts saved to {out_dir}/")