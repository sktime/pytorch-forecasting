from typing import Tuple
import torch
from torch import nn


class SeriesDecomposition(nn.Module):
    """Implements series decomposition using learnable moving averages."""

    def __init__(self, kernel_size: int):
        super(SeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=self.padding)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decomposes input series into trend and seasonal components.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)

        Returns:
            Tuple of (trend_component, seasonal_component)
        """
        batch_size, seq_len, n_features = x.shape
        x_reshaped = x.reshape(batch_size * n_features, 1, seq_len)
        trend = self.avg_pool(x_reshaped)
        trend = trend.reshape(batch_size, seq_len, n_features)
        seasonal = x - trend

        return trend, seasonal
