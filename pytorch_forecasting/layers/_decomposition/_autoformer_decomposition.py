import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoformerDecomposition(nn.Module):
    """Moving-average based series decomposition."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor):
        B, L, C = x.shape
        pad = (self.kernel_size - 1) // 2

        x_t = x.permute(0, 2, 1)
        x_t = F.pad(x_t, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(x_t, self.kernel_size, stride=1)
        trend = trend.permute(0, 2, 1)

        seasonal = x - trend
        return seasonal, trend
