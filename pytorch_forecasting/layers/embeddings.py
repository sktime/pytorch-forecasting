"""
Implementation of embedding layers from `nn.Module`.
"""

import math
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DataEmbedding_inverted(nn.Module):
    """
    Data embedding module for time series data.
    Args:
        c_in (int): Number of input features.
        d_model (int): Dimension of the model.
        embed_type (str): Type of embedding to use. Defaults to "fixed".
        freq (str): Frequency of the time series data. Defaults to "h".
        dropout (float): Dropout rate. Defaults to 0.1.
    """

    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    """
    Positional embedding module for time series data.
    Args:
        d_model (int): Dimension of the model.
        max_len (int): Maximum length of the input sequence. Defaults to 5000."""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


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
