"""
Data embedding layer for exogenous variables.
"""

import math
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_forecasting.layers._embeddings._temporal_embedding import TemporalEmbedding
from pytorch_forecasting.layers._embeddings._positional_embedding import PositionalEmbedding
from pytorch_forecasting.layers._embeddings._token_embedding import TokenEmbedding


class TimeFeatureEmbedding(nn.Module):
    """
    Embeds time-related features into the model dimension.

    Parameters
    ----------
    d_model : int
        Output embedding dimension.
    embed_type : str, optional
        Embedding type; kept for API compatibility (default: 'timeF').
    freq : str, optional
        Frequency string used to determine the number of input features
        (default: 'h').
    """

    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 1, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)
    

class DataEmbedding_wo_pos(nn.Module):
    """
    Data embedding that includes value and temporal embeddings but omits
    positional encodings.

    Parameters
    ----------
    c_in : int
        Number of input channels/features for the token embedding.
    d_model : int
        Dimension of the output embeddings.
    embed_type : str, optional
        Temporal embedding type passed to `TemporalEmbedding` (default: 'fixed').
    freq : str, optional
        Frequency string forwarded to temporal embedding or
        `TimeFeatureEmbedding` (default: 'h').
    dropout : float, optional
        Dropout probability applied to the output (default: 0.1).
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    """
    Data embedding module for time series data.

    Parameters
    ----------
    c_in : int
         Number of input features.
    d_model : int
        Dimension of the model.
    embed_type : str
        Type of embedding to use. Defaults to "fixed".
    freq : str
        Frequency of the time series data. Defaults to "h".
    dropout : float
        Dropout rate. Defaults to 0.1.
    """

    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
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
