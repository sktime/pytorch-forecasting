"""
Implementation of EncoderLayer for SOFTS from `nn.Module`.
"""

import torch
import torch.nn as nn

from pytorch_forecasting.layers._blocks._softs_block import STADModule


class SOFTSEncoderLayer(nn.Module):
    """
    Single encoder layer for SOFTS, combining STAD and a Feed-Forward Network.

    Applies Pre-LayerNorm STAD (cross-channel) then FFN (within-channel)
    with residual connections, following the Pre-LN Transformer convention.

    Parameters
    ----------
    d_model : int
        Embedding dimension per channel per time step.
    d_core : int
        Dimension of the central star node in the STAD sub-layer.
    d_ff : int
        Hidden dimension of the feed-forward network (typically 4 x d_model).
    dropout : float, default=0.0
        Dropout probability applied after the STAD and FFN sub-layers.
    """

    def __init__(self, d_model: int, d_core: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.stad = STADModule(d_model=d_model, d_core=d_core, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply one SOFTS encoder layer: STAD sub-layer then FFN sub-layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, n_channels, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch_size, n_channels, seq_len, d_model)``.
        """
        x = x + self.dropout(self.stad(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x
