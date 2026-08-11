"""
ModernTCN building blocks: ReparamLargeKernelConv, ModernTCNBlock, and Flatten_Head.
"""

import torch
import torch.nn as nn


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


class Flatten_Head(nn.Module):
    """
    Flatten Head.

    This layer flattens the input and projects
    it to the target window.

    Parameters
    ----------
    individual : bool
        If True, uses a separate linear projection per variable.
    n_vars : int
        Number of variables.
    nf : int
        Number of features.
    target_window : int
        Length of the target window.
    head_dropout : float
        Dropout rate.
    """

    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
