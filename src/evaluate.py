"""Shared evaluation helpers: predictions, confusion matrix, test-set scoring."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def plot_confusion(y_true, y_pred, classes, path, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 2, 1.1 * len(classes) + 1))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary(y_true, y_pred, classes) -> dict:
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    return {"accuracy": float((y_true == y_pred).mean()), "report": report}
