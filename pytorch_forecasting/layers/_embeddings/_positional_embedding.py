"""
Positional Embedding Layer for PTF.
"""

import math
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """
    Positional embedding module for time series data.

    Computes fixed sinusoidal positional encodings. Optionally adds the
    encoding to the input tensor and applies dropout.

    Parameters
    ----------
    d_model : int
        Dimension of the model.
    max_len : int
        Maximum length of the input sequence. Defaults to 5000.
    dropout : float
        Dropout probability applied after positional encoding.
        Defaults to 0.0 (no dropout).
    add_x : bool
        If ``True``, the forward pass returns ``dropout(x + pe)`` instead
        of just the raw positional encoding buffer. Defaults to ``False``.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.0, add_x=False):
        super().__init__()
        self.add_x = add_x
        self.drop = nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.add_x:
            return self.drop(x + self.pe[:, : x.size(1), :])
        return self.pe[:, : x.size(1)]
