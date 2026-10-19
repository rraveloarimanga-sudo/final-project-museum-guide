"""Feed-forward network, built from nn.Linear primitives (no high-level model magic)."""
from __future__ import annotations

from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}


class MLP(nn.Module):
    """input image -> flatten -> [Linear -> activation -> (Dropout)] x N -> Linear -> logits."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden=(512, 256),
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        act = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        d = in_features
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))
