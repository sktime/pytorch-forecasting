"""
Data embedding layer for exogenous variables.
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
