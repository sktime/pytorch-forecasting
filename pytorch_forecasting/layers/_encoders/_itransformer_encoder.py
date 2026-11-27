"""
Implementation of self-attention encoder layers for iTransformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class iTransformerEncoderLayer(nn.Module):
    """
    Self-attention encoder layer for the iTransformer model.

    Parameters
    ----------
    attention : nn.Module
        Self-attention mechanism (e.g., ``AttentionLayer`` wrapping ``FullAttention``).
    d_model : int
        Dimension of the model embeddings.
    d_ff : int, optional
        Dimension of the feed-forward network. Defaults to ``4 * d_model``.
    dropout : float, default=0.1
        Dropout rate.
    activation : str, default="gelu"
        Activation function (``"relu"`` or ``"gelu"``).
    """

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the encoder layer."""
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class iTransformerEncoder(nn.Module):
    """
    Stack of self-attention encoder layers for the iTransformer model.

    Parameters
    ----------
    attn_layers : list[nn.Module]
        List of ``iTransformerEncoderLayer`` instances.
    norm_layer : nn.Module, optional
        Optional final normalization layer (e.g., ``nn.LayerNorm``).
    """

    def __init__(
        self,
        attn_layers: list[nn.Module],
        norm_layer: nn.Module | None = None,
    ):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Forward pass of the encoder."""
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
