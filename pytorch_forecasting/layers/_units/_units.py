"""
Core Neural Network Layers for the UniTS architecture.
"""

import math

import torch
import torch.nn as nn


class _PatchEmbedding(nn.Module):
    """
    Project strided patches of a multivariate time series into d_model space.

    Uses channel-independent patching: each channel's patches are projected
    separately with a shared Linear(patch_len, d_model), then averaged across
    channels to match the UniTS paper's channel-independent approach.

    Parameters
    ----------
    patch_len : int
        Length of each patch window in time steps.
    stride : int
        Stride between consecutive patches.
    d_model : int
        Output embedding dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projection = nn.Linear(patch_len, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, n_channels).

        Returns
        -------
        torch.Tensor
            Shape (batch, num_patches, d_model).
        """
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        B, num_patches, C, P = patches.shape
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B * C, num_patches, P)
        emb = self.drop(self.projection(patches))
        emb = emb.view(B, C, num_patches, self.projection.out_features)

        # Channel independence: average across channels as per UniTS logic
        return emb.mean(dim=1)


class _PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int
        Maximum sequence length.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        half = d_model // 2
        div_term = torch.exp(
            torch.arange(0, half, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].size(1)])
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1), :])


class _TransformerBlock(nn.Module):
    """
    Pre-norm transformer encoder block (MHSA + FFN).

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x
