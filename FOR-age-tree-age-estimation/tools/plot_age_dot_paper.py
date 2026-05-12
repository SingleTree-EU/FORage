# plot_age_dots.py
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_and_plot(file_path, plot_path=None):
    df = pd.read_csv(file_path)

    true_ages = df["age_label"].values
    predicted_ages = (
        df["predicted_age"].values
        if "predicted_age" in df.columns
        else df["predicted_age_mean"].values
    )

    # -----------------------------
    # Metrics
    # -----------------------------
    mse = np.mean((predicted_ages - true_ages) ** 2)
    rmse = np.sqrt(mse)

    denom = np.sum((true_ages - true_ages.mean()) ** 2)
    r2 = 1 - np.sum((predicted_ages - true_ages) ** 2) / denom if denom > 0 else np.nan

    # -----------------------------
    # Figure
    # -----------------------------
    plt.figure(figsize=(7.5, 7.0))  

    # Axis range with padding
    min_val = min(true_ages.min(), predicted_ages.min())
    max_val = max(true_ages.max(), predicted_ages.max())
    pad = 0.03 * (max_val - min_val) if max_val > min_val else 1.0
    lo, hi = min_val - pad, max_val + pad

    # Scatter points (bigger & clearer)
    plt.scatter(
        true_ages,
        predicted_ages,
        s=120,                 
        alpha=0.7,
        facecolor="#4C72B0",
        edgecolor="white",
        linewidth=1.0,
        label="Data points"
    )

    # Line of equality
    plt.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=3.5,          
        color="#C44E52",
        label="Line of equality"
    )

    # Axis labels
    plt.xlabel("Reference age (years)", fontsize=22)
    plt.ylabel("Predicted age (years)", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)

    # Metrics box
    metrics_text = f"RMSE = {rmse:.2f} years\n" + r"$R^2$" + f" = {r2:.2f}"
    plt.text(
        0.04, 0.96,
        metrics_text,
        transform=plt.gca().transAxes,
        fontsize=20,         
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="0.6",
            alpha=0.95
        )
    )

    # Legend
    plt.legend(
        fontsize=18,
        frameon=False,
        loc="lower right"
    )

    # Grid
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.gca().set_axisbelow(True)

    plt.tight_layout()

    # -----------------------------
    # Save
    # -----------------------------
    if plot_path is None:
        plot_path = file_path.replace(".csv", ".png")

    if os.path.dirname(plot_path):
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {plot_path}")
    print(f"RMSE: {rmse:.2f}, R2: {r2:.2f}")

if __name__ == "__main__":
    evaluate_and_plot(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
