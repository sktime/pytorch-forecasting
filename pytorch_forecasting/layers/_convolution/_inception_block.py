"""Inception 2D convolution block used in TimesNet."""

import torch
import torch.nn as nn


class InceptionBlockV1(nn.Module):
    """Multi-scale 2D Inception block from TimesNet.

    Applies ``num_kernels`` parallel 2D convolutions with kernel sizes
    ``1, 3, 5, ..., 2 * num_kernels - 1`` and averages their outputs.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    num_kernels : int, default=6
        Number of parallel convolution kernels of increasing odd sizes.
    init_weight : bool, default=True
        Whether to initialize Conv2d weights with Kaiming normal initialization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
        init_weight: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * i + 1,
                    padding=i,
                )
                for i in range(self.num_kernels)
            ]
        )
        if init_weight:
            self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_list = [kernel(x) for kernel in self.kernels]
        return torch.stack(res_list, dim=-1).mean(-1)
