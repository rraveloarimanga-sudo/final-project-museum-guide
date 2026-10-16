"""Manual training loop with monitoring, written out so every number is visible.

Logs per epoch: train/val loss and accuracy, and (optionally) the global gradient norm —
the signal we use in the diagnosis step to show vanishing / exploding gradients.
"""
from __future__ import annotations

import copy

import torch
from torch import nn


def grad_global_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def run_epoch(model, loader, criterion, device, optimizer=None, log_grad=False):
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum, correct, total, grad_sum, steps = 0.0, 0, 0, 0.0, 0

    torch.set_grad_enabled(is_train)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if log_grad:
                grad_sum += grad_global_norm(model)
                steps += 1
            optimizer.step()
        loss_sum += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    torch.set_grad_enabled(True)

    out = {"loss": loss_sum / total, "acc": correct / total}
    if is_train and log_grad and steps:
        out["grad_norm"] = grad_sum / steps
    return out


def fit(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int | None = None,
    log_grad: bool = False,
):
    """Returns (history, best_state). History has one row per epoch."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "grad_norm")}
    best_val, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device, optimizer, log_grad)
        va = run_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        history["grad_norm"].append(tr.get("grad_norm"))
        print(
            f"epoch {epoch:2d} | train loss {tr['loss']:.3f} acc {tr['acc']:.3f} "
            f"| val loss {va['loss']:.3f} acc {va['acc']:.3f}"
        )

        if va["loss"] < best_val - 1e-4:
            best_val, best_state, bad = va["loss"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if patience is not None and bad >= patience:
                print(f"early stop at epoch {epoch} (no val improvement for {patience})")
                break

    model.load_state_dict(best_state)
    return history, best_state
