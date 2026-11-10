from __future__ import annotations

import torch.nn as nn

from .cnn import SimpleCNN, resnet
from .crnn import CRNN


def build_model(cfg: dict, num_classes: int) -> nn.Module:
    """Instantiate a model from the `model:` section of a config."""
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, **cfg)
    if name == "crnn":
        return CRNN(num_classes=num_classes, **cfg)
    if name.startswith("resnet"):
        return resnet(name, num_classes=num_classes, **cfg)
    raise ValueError(f"Unknown model: {name}")


__all__ = ["build_model", "SimpleCNN", "CRNN", "resnet"]
