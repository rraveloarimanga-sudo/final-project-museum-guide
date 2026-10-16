"""Small helpers: reproducibility, device, timing, metric logging."""
from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

from src.config import METRICS_DIR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    # The kiosk target is CPU; we train on CPU too so timings are honest.
    return torch.device("cpu")


@contextmanager
def timer(name: str, store: dict | None = None):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[time] {name}: {elapsed:.2f}s")
    if store is not None:
        store[name] = elapsed


def save_metrics(name: str, metrics: dict) -> None:
    path = METRICS_DIR / f"{name}.json"
    path.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"[saved] {path}")
