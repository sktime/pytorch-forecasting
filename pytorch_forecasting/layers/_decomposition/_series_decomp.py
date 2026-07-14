"""
Series Decomposition Block for time series forecasting models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_forecasting.layers._filter._moving_avg_filter import MovingAvg


class SeriesDecomposition(nn.Module):
    """
    Series decomposition block from Autoformer.

    Decomposes time series into trend and seasonal components using
    moving average filtering.

    Parameters
    ----------
    kernel_size : int
        Size of the moving average kernel for trend extraction.
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        """
        Decompose input time series into trend and seasonal components.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, features) containing
            the time series data.

        Returns
        -------
        seasonal : torch.Tensor
            Seasonal component (residual after trend removal) with same shape
            as input.
        trend : torch.Tensor
            Trend component extracted via moving average with same shape as
            input.
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend
