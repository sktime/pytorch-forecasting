"""
Moving Average Filter Block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvg(nn.Module):
    """
    Moving Average block for smoothing and trend extraction from time series data.

    A moving average is a smoothing technique that creates a series of average from
    different subsets of a time series.

    For example: Given a time series ``x = [x_1, x_2, ..., x_n]``, the moving average
    with a kernel size of `k` calculates the average of each subset of `k` consecutive
    elements, resulting in a new series of averages.

    Args:
        kernel_size (int):
            Size of the moving average kernel.
        stride (int):
            Stride for the moving average operation, typically set to 1.
    """

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=stride, padding=0)

    def forward(self, x):
        if self.kernel_size % 2 == 0:
            self.padding_left = self.kernel_size // 2 - 1
            self.padding_right = self.kernel_size // 2
        else:
            self.padding_left = self.kernel_size // 2
            self.padding_right = self.kernel_size // 2

        front = x[:, 0:1, :].repeat(1, self.padding_left, 1)
        end = x[:, -1:, :].repeat(1, self.padding_right, 1)

        x_padded = torch.cat([front, x, end], dim=1)
        x_transposed = x_padded.permute(0, 2, 1)
        x_smoothed = self.avg(x_transposed)
        x_out = x_smoothed.permute(0, 2, 1)
        return x_out
