"""
Implementation of output layers from `nn.Module` for TimeXer model.
"""

import math
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlattenHead(nn.Module):
    """
    Flatten head for the output of the model.
    Args:
        n_vars (int): Number of input features.
        nf (int): Number of features in the last layer.
        target_window (int): Target window size.
        head_dropout (float): Dropout rate for the head. Defaults to 0.
        n_quantiles (int, optional): Number of quantiles. Defaults to None."""

    def __init__(self, n_vars, nf, target_window, head_dropout=0, n_quantiles=None):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.n_quantiles = n_quantiles

        if self.n_quantiles is not None:
            self.linear = nn.Linear(nf, target_window * n_quantiles)
        else:
            self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)

        if self.n_quantiles is not None:
            batch_size, n_vars = x.shape[0], x.shape[1]
            x = x.reshape(batch_size, n_vars, -1, self.n_quantiles)
        return x
