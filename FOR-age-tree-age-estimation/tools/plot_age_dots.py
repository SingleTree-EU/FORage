# plot_age_dots.py
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

def evaluate_and_plot(file_path, plot_path=None):
    df = pd.read_csv(file_path)
    true_ages = df["age_label"]
    predicted_ages = df["predicted_age"] if "predicted_age" in df.columns else df["predicted_age_mean"]

    mae = (predicted_ages - true_ages).abs().mean()
    mse = ((predicted_ages - true_ages) ** 2).mean()
    rmse = mse ** 0.5
    r2 = 1 - ((predicted_ages - true_ages) ** 2).sum() / ((true_ages - true_ages.mean()) ** 2).sum()
    mbe = (predicted_ages - true_ages).mean()

    plt.figure(figsize=(8, 6))
    plt.scatter(true_ages, predicted_ages, alpha=0.7)
    plt.plot([true_ages.min(), true_ages.max()], [true_ages.min(), true_ages.max()], 'r--')
    plt.xlabel("True age")
    plt.ylabel("Predicted age")
    plt.title("Predicted vs True age")
    metrics_text = f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}\nMBE: {mbe:.2f}"
    plt.gca().annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction',
                       fontsize=12, bbox=dict(facecolor='white', alpha=0.7),
                       verticalalignment='top', horizontalalignment='left')
    plt.grid(True)

    if plot_path is None:
        plot_path = file_path.replace(".csv", ".png")

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")

if __name__ == "__main__":
    evaluate_and_plot(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
