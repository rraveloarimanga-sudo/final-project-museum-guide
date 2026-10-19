"""Step 2 — Feed-forward network in PyTorch.

Trains the MLP on 64x64 images, logs train/val curves, saves a confusion matrix and
the final test accuracy. This is the deliverable "a feed-forward network for artwork
classification" plus "training monitoring".
Run: python scripts/02_mlp.py
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, FIG_DIR, MLP_IMG_SIZE, NUM_WORKERS, SEED
from src.data import load_dataframe, make_datasets
from src.evaluate import plot_confusion, predict, summary
from src.mlp import MLP
from src.transforms import mlp_transforms
from src.utils import get_device, save_metrics, set_seed, timer


def plot_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(epochs, history["train_loss"], label="train")
    a1.plot(epochs, history["val_loss"], label="val")
    a1.set_title("loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(epochs, history["train_acc"], label="train")
    a2.plot(epochs, history["val_acc"], label="val")
    a2.set_title("accuracy"); a2.set_xlabel("epoch"); a2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    set_seed(SEED)
    device = get_device()
    df = load_dataframe()
    transforms = {"train": mlp_transforms(True), "val": mlp_transforms(False),
                  "test": mlp_transforms(False)}
    datasets, classes = make_datasets(df, transforms, cache_size=MLP_IMG_SIZE)
    loaders = {k: DataLoader(v, batch_size=BATCH_SIZE, shuffle=(k == "train"),
                             num_workers=NUM_WORKERS) for k, v in datasets.items()}

    in_features = MLP_IMG_SIZE * MLP_IMG_SIZE * 3
    model = MLP(in_features, len(classes), hidden=(512, 256), dropout=0.0).to(device)

    # Deferred import so the file reads top-down with the config above.
    from src.train import fit

    timings: dict = {}
    with timer("mlp_train", timings):
        history, _ = fit(model, loaders["train"], loaders["val"], device,
                         epochs=40, lr=1e-3, weight_decay=0.0, patience=8, log_grad=True)

    plot_curves(history, FIG_DIR / "mlp_curves.png")
    y_true, y_pred = predict(model, loaders["test"], device)
    plot_confusion(y_true, y_pred, classes, FIG_DIR / "mlp_confusion_test.png",
                   "MLP — test")

    metrics = {
        "classes": classes,
        "in_features": in_features,
        "n_params": sum(p.numel() for p in model.parameters()),
        "history": history,
        "test": summary(y_true, y_pred, classes),
        "timings_s": timings,
    }
    print(f"[mlp] test accuracy = {metrics['test']['accuracy']:.3f}")
    save_metrics("mlp", metrics)


if __name__ == "__main__":
    main()
