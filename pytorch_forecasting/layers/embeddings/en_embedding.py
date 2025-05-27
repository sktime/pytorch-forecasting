"""
Implementation of endogenous embedding layers from `nn.Module`.
"""

import math
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_forecasting.layers.embeddings import PositionalEmbedding


class EnEmbedding(nn.Module):
    """
    Encoder embedding module for time series data. Handles endogenous feature
    embeddings in this case.
    Args:
        n_vars (int): Number of input features.
        d_model (int): Dimension of the model.
        patch_len (int): Length of the patches.
        dropout (float): Dropout rate. Defaults to 0.1.
    """

    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()

        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars
