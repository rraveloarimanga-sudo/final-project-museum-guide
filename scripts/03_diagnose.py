"""Step 3 — Training monitoring and failure diagnosis.

Three controlled experiments, every claim backed by a saved number/figure:

  1. Learning-rate sweep   : too high diverges, too low underfits, a middle value is best.
  2. Activation & gradients: sigmoid in a deeper MLP shrinks gradients (vanishing) vs ReLU.
  3. Overfitting & fix     : a big MLP trained long overfits (train >> val); dropout + weight
                             decay close the gap.

Images are cached once in memory, so the many short runs stay fast on CPU.
Run: python scripts/03_diagnose.py
"""
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, FIG_DIR, MLP_IMG_SIZE, NUM_WORKERS, SEED
from src.data import load_dataframe, make_datasets
from src.mlp import MLP
from src.train import run_epoch
from src.transforms import mlp_transforms
from src.utils import get_device, save_metrics, set_seed

IN = MLP_IMG_SIZE * MLP_IMG_SIZE * 3


def train_run(model, loaders, device, epochs, lr, weight_decay=0.0, log_grad=False):
    """Minimal loop built on run_epoch, so we control lr / wd / gradient logging."""
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "grad_norm")}
    for _ in range(epochs):
        tr = run_epoch(model, loaders["train"], criterion, device, opt, log_grad)
        va = run_epoch(model, loaders["val"], criterion, device)
        hist["train_loss"].append(tr["loss"]); hist["train_acc"].append(tr["acc"])
        hist["val_loss"].append(va["loss"]); hist["val_acc"].append(va["acc"])
        hist["grad_norm"].append(tr.get("grad_norm"))
    return hist


def diverged(hist) -> bool:
    last = hist["train_loss"][-1]
    return math.isnan(last) or math.isinf(last) or last > hist["train_loss"][0] * 1.5


# --------------------------------------------------------------------------------------------------
def exp_learning_rate(loaders, classes, device):
    lrs = [1e-1, 1e-2, 1e-3, 1e-4]
    results, fig = {}, plt.figure(figsize=(7, 5))
    for lr in lrs:
        set_seed(SEED)
        model = MLP(IN, len(classes), hidden=(512, 256)).to(device)
        hist = train_run(model, loaders, device, epochs=25, lr=lr)
        results[f"{lr:g}"] = {
            "final_train_acc": hist["train_acc"][-1],
            "best_val_acc": max(hist["val_acc"]),
            "diverged": diverged(hist),
        }
        plt.plot(range(1, len(hist["val_loss"]) + 1), hist["val_loss"], label=f"lr={lr:g}")
        print(f"[lr {lr:g}] best val acc {max(hist['val_acc']):.3f}"
              f"{'  (DIVERGED)' if diverged(hist) else ''}")
    plt.xlabel("epoch"); plt.ylabel("val loss"); plt.title("Learning-rate sweep")
    plt.legend(); plt.tight_layout()
    fig.savefig(FIG_DIR / "diag_lr_sweep.png", dpi=150); plt.close(fig)
    return results


def exp_activation_gradients(loaders, classes, device):
    """Deeper net to expose vanishing gradients with sigmoid vs ReLU."""
    results, fig = {}, plt.figure(figsize=(7, 5))
    for act in ("relu", "sigmoid"):
        set_seed(SEED)
        model = MLP(IN, len(classes), hidden=(512, 256, 128, 64), activation=act).to(device)
        hist = train_run(model, loaders, device, epochs=25, lr=1e-3, log_grad=True)
        grads = [g for g in hist["grad_norm"] if g is not None]
        results[act] = {
            "best_val_acc": max(hist["val_acc"]),
            "mean_grad_norm": sum(grads) / len(grads),
            "final_grad_norm": grads[-1],
        }
        plt.plot(range(1, len(grads) + 1), grads, label=act)
        print(f"[{act}] best val acc {max(hist['val_acc']):.3f} | mean grad norm {results[act]['mean_grad_norm']:.4f}")
    plt.xlabel("epoch"); plt.ylabel("mean gradient norm"); plt.yscale("log")
    plt.title("Gradient norm: ReLU vs sigmoid (deep MLP)"); plt.legend(); plt.tight_layout()
    fig.savefig(FIG_DIR / "diag_gradients.png", dpi=150); plt.close(fig)
    return results


def exp_overfitting(loaders, classes, device):
    """Big MLP, long training: without regularisation it overfits; dropout + wd close the gap."""
    settings = {"no_reg": dict(dropout=0.0, weight_decay=0.0),
                "with_reg": dict(dropout=0.5, weight_decay=1e-3)}
    results, fig = {}, None
    curves = {}
    for name, cfg in settings.items():
        set_seed(SEED)
        model = MLP(IN, len(classes), hidden=(1024, 512), dropout=cfg["dropout"]).to(device)
        hist = train_run(model, loaders, device, epochs=60, lr=1e-3, weight_decay=cfg["weight_decay"])
        gap = hist["train_acc"][-1] - hist["val_acc"][-1]
        results[name] = {
            "final_train_acc": hist["train_acc"][-1],
            "final_val_acc": hist["val_acc"][-1],
            "best_val_acc": max(hist["val_acc"]),
            "train_val_gap": gap,
        }
        curves[name] = hist
        print(f"[{name}] train {hist['train_acc'][-1]:.3f} val {hist['val_acc'][-1]:.3f} gap {gap:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name in zip(axes, settings):
        h = curves[name]
        ep = range(1, len(h["train_acc"]) + 1)
        ax.plot(ep, h["train_acc"], label="train")
        ax.plot(ep, h["val_acc"], label="val")
        ax.set_title(name); ax.set_xlabel("epoch"); ax.legend()
    axes[0].set_ylabel("accuracy")
    fig.suptitle("Overfitting without vs with regularisation")
    fig.tight_layout(); fig.savefig(FIG_DIR / "diag_overfitting.png", dpi=150); plt.close(fig)
    return results


def main():
    set_seed(SEED)
    device = get_device()
    df = load_dataframe()
    transforms = {"train": mlp_transforms(True), "val": mlp_transforms(False),
                  "test": mlp_transforms(False)}
    datasets, classes = make_datasets(df, transforms, cache_size=MLP_IMG_SIZE)
    loaders = {k: DataLoader(v, batch_size=BATCH_SIZE, shuffle=(k == "train"),
                             num_workers=NUM_WORKERS) for k, v in datasets.items()}

    metrics = {
        "learning_rate": exp_learning_rate(loaders, classes, device),
        "activation_gradients": exp_activation_gradients(loaders, classes, device),
        "overfitting": exp_overfitting(loaders, classes, device),
    }
    save_metrics("diagnose", metrics)


if __name__ == "__main__":
    main()
