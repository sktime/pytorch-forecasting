"""
Implementation of the recursive SCI tree encoder for SCINet from `nn.Module`.
"""

import torch
import torch.nn as nn

from pytorch_forecasting.layers._blocks._scinet_block import SCIBlock


class SCITree(nn.Module):
    """Recursive binary tree of SCI-Blocks.

    At each level the sequence is split into two halves; each half is
    processed by a child ``SCITree`` of depth ``num_levels - 1``.
    The outputs are interleaved back into a sequence of the original
    length before being returned.

    Parameters
    ----------
    n_channels : int
        Number of feature channels.
    num_levels : int
        Depth of the binary decomposition tree (>= 1).
    hid_size : int, default=1
        Channel expansion factor forwarded to every ``SCIBlock``.
    kernel_size : int, default=5
        Kernel width forwarded to every ``SCIBlock``.
    dropout : float, default=0.5
        Dropout probability forwarded to every ``SCIBlock``.
    """

    def __init__(
        self,
        n_channels: int,
        num_levels: int,
        hid_size: int = 1,
        kernel_size: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.block = SCIBlock(n_channels, hid_size, kernel_size, dropout)

        if num_levels > 1:
            self.even_tree = SCITree(
                n_channels, num_levels - 1, hid_size, kernel_size, dropout
            )
            self.odd_tree = SCITree(
                n_channels, num_levels - 1, hid_size, kernel_size, dropout
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recursively decompose, transform, and reconstruct the sequence.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, C)`` where ``T`` must be divisible by
            ``2 ** num_levels``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, C)``.
        """
        even_out, odd_out = self.block(x)

        if self.num_levels > 1:
            even_out = self.even_tree(even_out)
            odd_out = self.odd_tree(odd_out)

        # Interleave even and odd back into original order
        B, T_half, C = even_out.shape
        out = torch.empty(B, T_half * 2, C, device=x.device, dtype=x.dtype)
        out[:, 0::2, :] = even_out
        out[:, 1::2, :] = odd_out
        return out
