"""
Reparameterizable Large Kernel Convolution.
"""

import torch
import torch.nn as nn


class ReparamLargeKernelConv(nn.Module):
    """
    Reparameterizable Large Kernel Convolution.

    This layer uses a large kernel (kernel_size) and
    a small kernel in parallel,then adds their outputs.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Large kernel size.
    stride : int
        Stride.
    groups : int
        Number of groups.
    small_kernel_size : int
        Small kernel size.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, groups, small_kernel_size
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel_size = small_kernel_size

        padding = kernel_size // 2
        self.lkb_origin = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )

        self.small_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=small_kernel_size,
                stride=stride,
                padding=small_kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.lkb_origin(x) + self.small_conv(x)
