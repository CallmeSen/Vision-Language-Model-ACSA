"""
Plot train_loss vs dev_loss from train_result.json
Usage: python plot_loss.py [--dir outputs/]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="outputs", help="Output directory containing train_result.json")
    parser.add_argument("--save", type=str, default=None, help="Save figure to path")
    args = parser.parse_args()

    result_path = Path(args.dir) / "train_result.json"
    if not result_path.exists():
        print(f"ERROR: {result_path} not found. Run training first.")
        return

    with open(result_path) as f:
        data = json.load(f)
    epochs_data = data if isinstance(data, list) else data.get("epochs", data)

    if not epochs_data:
        print("train_result.json has no epoch data.")
        return

    epochs = [r["epoch"] for r in epochs_data]
    train_losses = [r["train_loss"] for r in epochs_data]
    dev_losses = [r["dev_loss"] for r in epochs_data]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, "o-", color="#2196F3", label="Train Loss", linewidth=2, markersize=6)
    ax.plot(epochs, dev_losses, "s-", color="#F44336", label="Dev Loss", linewidth=2, markersize=6)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training vs Dev Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate min dev loss
    min_dev_idx = dev_losses.index(min(dev_losses))
    ax.annotate(
        f"Min: {dev_losses[min_dev_idx]:.4f}",
        xy=(epochs[min_dev_idx], dev_losses[min_dev_idx]),
        xytext=(8, -16), textcoords="offset points",
        fontsize=9, color="#F44336",
        arrowprops=dict(arrowstyle="->", color="#F44336", lw=1),
    )

    # Best F1 annotation
    best_f1 = data.get("best_f1", None) if isinstance(data, dict) else None
    if best_f1 is not None:
        ax.text(0.02, 0.98, f"Best Dev F1: {best_f1:.4f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
