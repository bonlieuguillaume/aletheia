"""CNN classifiers for single-channel SAR images."""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """Small from-scratch baseline. No early downsampling, so fine
    high-frequency (forensic) traces survive the first layers."""

    def __init__(self, num_classes: int = 2, in_chans: int = 1, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        layers, c = [], in_chans
        for w in widths:
            layers.append(conv_block(c, w))
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c, num_classes),
        )
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def resnet(name: str = "resnet18", num_classes: int = 2, in_chans: int = 1, pretrained: bool = True) -> nn.Module:
    """torchvision ResNet adapted to 1-channel input.

    ImageNet RGB weights of conv1 are averaged into a single channel.
    """
    import torchvision

    weights = "DEFAULT" if pretrained else None
    model = getattr(torchvision.models, name)(weights=weights)
    old = model.conv1
    model.conv1 = nn.Conv2d(in_chans, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    with torch.no_grad():
        model.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
