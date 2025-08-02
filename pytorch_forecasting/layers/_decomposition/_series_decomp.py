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

    Args:
        kernel_size (int):
            Size of the moving average kernel for trend extraction.
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        """
        Forward pass for series decomposition.

        Args:
            x (torch.Tensor):
                Input time series tensor of shape (batch_size, seq_len, features).

        Returns:
            tuple:
                - trend (torch.Tensor): Trend component of the time series.
                - seasonal (torch.Tensor): Seasonal component of the time series.
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend
