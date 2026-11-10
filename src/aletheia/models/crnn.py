"""CNN + RNN (CRNN) classifier.

A CNN turns the image into a grid of local feature vectors; a bidirectional
LSTM/GRU then reads that grid as a sequence (raster order), so it can model
*spatial consistency* between regions — a spliced/inpainted area tends to
break the statistics shared by the rest of the scene.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cnn import conv_block


class CRNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        in_chans: int = 1,
        widths=(32, 64, 128, 256),
        grid: int = 8,
        rnn_type: str = "lstm",
        hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers, c = [], in_chans
        for w in widths:
            layers.append(conv_block(c, w))
            c = w
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(grid)  # -> (B, C, grid, grid)

        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.rnn = rnn_cls(
            input_size=c,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.pool(self.cnn(x))  # (B, C, g, g)
        seq = f.flatten(2).transpose(1, 2)  # (B, g*g, C)
        out, _ = self.rnn(seq)  # (B, g*g, 2*hidden)
        # mean captures global statistics, max catches a single odd region
        pooled = torch.cat([out.mean(dim=1), out.amax(dim=1)], dim=1)
        return self.head(pooled)
