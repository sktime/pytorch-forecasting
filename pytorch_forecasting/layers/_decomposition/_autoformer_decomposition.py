"""
Autoformer Series Decomposition with enhanced moving average.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoformerDecomposition(nn.Module):
    """
    
    Decomposes time series into trend and seasonal components using
    a moving average filter.
    
    Args:
        kernel_size (int): Size of the moving average kernel for trend extraction.
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Decompose time series into trend and seasonal components.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, features]
        
        Returns:
            tuple:
                - seasonal (torch.Tensor): Seasonal component [batch, seq_len, features]
                - trend (torch.Tensor): Trend component [batch, seq_len, features]
        """
        batch, length, channels = x.shape
        
        x_permuted = x.permute(0, 2, 1)
        
        
        padding = (self.kernel_size - 1) // 2
        
        
        x_padded = F.pad(x_permuted, (padding, padding), mode='replicate')
        
        trend = F.avg_pool1d(
            x_padded,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0
        )
        
      
        trend = trend.permute(0, 2, 1)
        
       
        seasonal = x - trend
        
        return seasonal, trend