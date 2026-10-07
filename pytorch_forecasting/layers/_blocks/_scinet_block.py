"""
Implementation of the SCI-Block for SCINet from `nn.Module`.

Implements the sample-convolution-and-interaction block of `SCINet: Time
Series Modeling and Forecasting with Sample Convolution and Interaction
Networks <https://arxiv.org/abs/2106.09305>`_ by Liu et al. (NeurIPS 2022),
adapted from the authors' original implementation at
`cure-lab/SCINet <https://github.com/cure-lab/SCINet>`_.
"""

import torch
import torch.nn as nn


class SCIBlock(nn.Module):
    """Single Sample-Convolution-and-Interaction block.

    Splits the input sequence into even- and odd-indexed sub-sequences,
    applies four distinct convolutional modules (phi, psi, rho, eta) to
    produce interactive, bi-directionally modulated outputs.

    Parameters
    ----------
    n_channels : int
        Number of input feature channels (C).
    hid_size : int, default=1
        Channel expansion factor for the hidden convolution layer.
        Hidden channels = n_channels * hid_size.
    kernel_size : int, default=5
        Kernel width for all Conv1d layers.
    dropout : float, default=0.5
        Dropout probability inside each conv module.
    """

    def __init__(
        self,
        n_channels: int,
        hid_size: int = 1,
        kernel_size: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        hid_channels = max(1, n_channels * hid_size)
        self.phi = _make_conv_module(n_channels, hid_channels, kernel_size, dropout)
        self.psi = _make_conv_module(n_channels, hid_channels, kernel_size, dropout)
        self.rho = _make_conv_module(n_channels, hid_channels, kernel_size, dropout)
        self.eta = _make_conv_module(n_channels, hid_channels, kernel_size, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the SCI interaction to even and odd sub-sequences.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, C)``.

        Returns
        -------
        even_out : torch.Tensor
            Shape ``(B, T//2, C)``.
        odd_out : torch.Tensor
            Shape ``(B, T//2, C)``.
        """
        even = x[:, 0::2, :]  # (B, T/2, C)
        odd = x[:, 1::2, :]  # (B, T/2, C)

        # Transpose to (B, C, T/2) for Conv1d
        even_t = even.permute(0, 2, 1)
        odd_t = odd.permute(0, 2, 1)

        # Multiplicative interaction
        even_scaled = even * torch.exp(self.phi(odd_t).permute(0, 2, 1))
        odd_scaled = odd * torch.exp(self.psi(even_t).permute(0, 2, 1))

        # Additive interaction
        even_out = even_scaled + self.rho(odd_scaled.permute(0, 2, 1)).permute(0, 2, 1)
        odd_out = odd_scaled + self.eta(even_scaled.permute(0, 2, 1)).permute(0, 2, 1)

        return even_out, odd_out


def _make_conv_module(
    in_channels: int,
    hid_channels: int,
    kernel_size: int,
    dropout: float,
) -> nn.Sequential:
    """Build a single conv sub-module used inside an SCI-Block.

    Parameters
    ----------
    in_channels : int
        Number of input (and output) channels.
    hid_channels : int
        Intermediate channel width after the first convolution.
    kernel_size : int
        Kernel width for both Conv1d layers.
    dropout : float
        Dropout probability applied between the two convolutions.

    Returns
    -------
    nn.Sequential
        Conv1d -> LeakyReLU -> Dropout -> Conv1d -> Tanh pipeline.
    """
    pad = kernel_size // 2
    return nn.Sequential(
        nn.ReplicationPad1d(pad),
        nn.Conv1d(in_channels, hid_channels, kernel_size),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
        nn.ReplicationPad1d(pad),
        nn.Conv1d(hid_channels, in_channels, kernel_size),
        nn.Tanh(),
    )
