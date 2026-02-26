"""Data embedding utilities."""

import math
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_forecasting.layers._embeddings._positional_embedding import (
    PositionalEmbedding,
)
from pytorch_forecasting.layers._embeddings._temporal_embedding import TemporalEmbedding
from pytorch_forecasting.layers._embeddings._token_embedding import TokenEmbedding


class TimeFeatureEmbedding(nn.Module):
    """Embed numeric time features into the model dimension.

    Args:
        d_model (int): output embedding dimension.
        embed_type (str): unused but kept for API compatibility.
        freq (str): frequency code determines the expected number of input
            time features (e.g., 'h' -> 1, 't' -> 5).
    """

    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super().__init__()

        freq_map = {"h": 1, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        """Apply a linear projection to the input time features.

        Args:
            x (torch.Tensor): tensor of numeric time features with last
                dimension equal to the number of features implied by
                `freq`.

        Returns:
            torch.Tensor: projected tensor with last dimension `d_model`.
        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        """Compose token, positional and temporal embeddings.

        Args:
            c_in (int): number of input features/channels for the token
                embedding.
            d_model (int): model/embedding dimensionality.
            embed_type (str): type of temporal embedding ('fixed', 'learned',
                or 'timeF').
            freq (str): frequency code passed to temporal embedding.
            dropout (float): dropout probability applied to summed embedding.
        """
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """Compute the full data embedding used by sequence models.

        Args:
            x (torch.Tensor): token/value input of shape `[batch, seq_len, c_in]`.
            x_mark (torch.Tensor or None): temporal marker tensor used to
                compute temporal embeddings (shape `[batch, seq_len, ...]`).

        Returns:
            torch.Tensor: embedded tensor of shape `[batch, seq_len, d_model]`.
        """
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
        return self.dropout(x)


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
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """Embed inputs where time is the last dimension.

        This variant expects `x` shaped `[batch, seq_len, c_in]` and performs a
        transpose before applying a linear projection. If `x_mark` is provided
        it is concatenated along the feature dimension before projection.

        Args:
            x (torch.Tensor): input tensor of shape `[batch, seq_len, c_in]`.
            x_mark (torch.Tensor or None): optional temporal features with the
                same leading shape as `x`.

        Returns:
            torch.Tensor: embedded tensor with dropout applied.
        """
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)
