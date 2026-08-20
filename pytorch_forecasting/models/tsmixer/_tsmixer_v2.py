"""
TSMixer model for PyTorch Forecasting.
-------------------------------------------
"""

from typing import Any
import warnings

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class TSMixerBlock(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        num_features: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Linear(sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sequence_length),
            nn.Dropout(dropout),
        )

        self.channel = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x
