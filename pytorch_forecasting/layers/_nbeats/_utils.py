"""
Utility functions for N-BEATS model implementation.
"""

import numpy as np
import torch.nn as nn


def linear(input_size, output_size, bias=True, dropout: int = None):
    """
    Initialize linear layers for MLP block layers.
    """
    lin = nn.Linear(input_size, output_size, bias=bias)
    if dropout is not None:
        return nn.Sequential(nn.Dropout(dropout), lin)
    else:
        return lin


def linspace(
    backcast_length: int, forecast_length: int, centered: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate linear spaced values for backcast and forecast.
    """
    if centered:
        norm = max(backcast_length, forecast_length)
        start = -backcast_length
        stop = forecast_length - 1
    else:
        norm = backcast_length + forecast_length
        start = 0
        stop = backcast_length + forecast_length - 1
    lin_space = np.linspace(
        start / norm, stop / norm, backcast_length + forecast_length, dtype=np.float32
    )
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls
