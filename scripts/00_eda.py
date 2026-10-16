"""Step 0 — Exploratory data analysis.

Produces the evidence for slide "What data do we have?":
class balance (both targets), image size distribution, a sample grid.
Run: python scripts/00_eda.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from src.config import FIG_DIR, SEED, TARGET
from src.data import encode_labels, load_dataframe, open_image, split_indices
from src.utils import save_metrics, set_seed


def plot_balance(df, column):
    counts = df[column].value_counts()
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(counts) + 1))
    counts.sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_xlabel("number of artworks")
    ax.set_title(f"Class balance — {column}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"eda_balance_{column}.png", dpi=150)
    plt.close(fig)
    return counts


def main():
    set_seed(SEED)
    df = load_dataframe()
    print(df.head())
    print(f"[eda] {len(df)} artworks, columns: {list(df.columns)}")

    metrics = {"n_artworks": len(df), "target": TARGET}
    for column in ("style", "genre"):
        if column in df.columns:
            counts = plot_balance(df, column)
            metrics[f"{column}_counts"] = counts.to_dict()
            metrics[f"{column}_imbalance_ratio"] = counts.max() / counts.min()
            # Majority-class accuracy: the floor every model must beat
            metrics[f"{column}_majority_baseline_acc"] = counts.max() / counts.sum()

    # Image sizes on a sample (opening every file is slow on CPU)
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(df), size=min(300, len(df)), replace=False)
    sizes = np.array([open_image(df["image"].iloc[i]).size for i in sample])
    metrics["img_width_median"] = float(np.median(sizes[:, 0]))
    metrics["img_height_median"] = float(np.median(sizes[:, 1]))

    # Sample grid: 4 examples per class (first 6 classes)
    classes = sorted(df[TARGET].unique())[:6]
    fig, axes = plt.subplots(len(classes), 4, figsize=(8, 2 * len(classes)))
    for r, c in enumerate(classes):
        picks = df[df[TARGET] == c].sample(4, random_state=SEED)
        for k, item in enumerate(picks["image"]):
            ax = axes[r, k]
            ax.imshow(open_image(item))
            ax.axis("off")
            if k == 0:
                ax.set_title(c, loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_samples.png", dpi=120)
    plt.close(fig)

    y, _ = encode_labels(df)
    metrics["split_sizes"] = {k: len(v) for k, v in split_indices(y).items()}
    save_metrics("eda", metrics)


if __name__ == "__main__":
    main()
