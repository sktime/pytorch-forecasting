"""
ModernTCN Block: For Modern Temporal Convolutional Network
"""

import torch
import torch.nn as nn

from pytorch_forecasting.layers._convolution._reparam_large_kernel_conv import (
    ReparamLargeKernelConv,
)


class ModernTCNBlock(nn.Module):
    """
    Modern TCN Block.

    This block is a residual block that consists of a depthwise
    separable convolution and a feed-forward network.

    Parameters
    ----------
    d_model : int
        Dimension of the model.
    kernel_size : int
        Size of the large kernel.
    small_kernel_size : int
        Size of the small kernel.
    d_ff : int
        Dimension of the feed-forward network.
    nvars : int
        Number of variables.
    dropout : float
        Dropout rate.
    """

    def __init__(self, d_model, kernel_size, small_kernel_size, d_ff, nvars, dropout):
        super().__init__()
        self.d_model = d_model
        self.nvars = nvars

        self.dwconv = ReparamLargeKernelConv(
            in_channels=nvars * d_model,
            out_channels=nvars * d_model,
            kernel_size=kernel_size,
            stride=1,
            groups=nvars * d_model,
            small_kernel_size=small_kernel_size,
        )
        self.norm = nn.BatchNorm1d(d_model)

        self.ffn1_pw1 = nn.Conv1d(
            nvars * d_model, nvars * d_ff, kernel_size=1, groups=nvars
        )
        self.ffn1_act = nn.GELU()
        self.ffn1_pw2 = nn.Conv1d(
            nvars * d_ff, nvars * d_model, kernel_size=1, groups=nvars
        )
        self.ffn1_drop1 = nn.Dropout(dropout)
        self.ffn1_drop2 = nn.Dropout(dropout)

        self.ffn2_pw1 = nn.Conv1d(
            nvars * d_model, nvars * d_ff, kernel_size=1, groups=d_model
        )
        self.ffn2_act = nn.GELU()
        self.ffn2_pw2 = nn.Conv1d(
            nvars * d_ff, nvars * d_model, kernel_size=1, groups=d_model
        )
        self.ffn2_drop1 = nn.Dropout(dropout)
        self.ffn2_drop2 = nn.Dropout(dropout)

    def forward(self, x):
        input_x = x
        B, M, D, N = x.shape

        x = x.reshape(B, M * D, N)
        x = self.dwconv(x)

        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M * D, N)

        x = self.ffn1_drop1(self.ffn1_pw1(x))
        x = self.ffn1_act(x)
        x = self.ffn1_drop2(self.ffn1_pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2_drop1(self.ffn2_pw1(x))
        x = self.ffn2_act(x)
        x = self.ffn2_drop2(self.ffn2_pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        return input_x + x
