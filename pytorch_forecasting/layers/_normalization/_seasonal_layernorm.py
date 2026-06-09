"""
Seasonal LayerNorm for the Autoformer architecture.

Implements a special layer normalization designed for the seasonal component
of decomposed time series, which removes the mean bias after normalization.
"""

import torch
import torch.nn as nn


class SeasonalLayerNorm(nn.Module):
    """
    Special designed layernorm for the seasonal part.
    """

    def __init__(self, channels):
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias
