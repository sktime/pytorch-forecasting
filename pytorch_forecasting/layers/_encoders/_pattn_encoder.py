"""
Self-attention encoder layer for PAttn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PAttnEncoderLayer(nn.Module):
    """
    Self-attention encoder layer with 1D pointwise feed-forward projections.

    Parameters
    ----------
    attention : nn.Module
        Self-attention mechanism returning ``(output, attention_weights)``.
    d_model : int
        Model embedding dimension.
    d_ff : int, optional
        Hidden dimension of the feed-forward network. Defaults to ``4 * d_model``.
    dropout : float, default=0.1
        Dropout probability applied after attention and feed-forward projections.
    activation : str, default="gelu"
        Activation function for the feed-forward block (``"relu"`` or ``"gelu"``).
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
        if activation not in ("relu", "gelu"):
            raise ValueError(
                f"activation must be 'relu' or 'gelu', got {activation!r}."
            )
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
        """
        Apply self-attention and pointwise feed-forward transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, seq_len, d_model)``.
        attn_mask : torch.Tensor, optional
            Optional attention mask passed to ``self.attention``.
        tau : torch.Tensor, optional
            Optional temporal scaling parameter for attention.
        delta : torch.Tensor, optional
            Optional delta parameter for attention.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Encoded tensor of shape ``(batch_size, seq_len, d_model)`` and
            optional attention weights.
        """
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = self.norm1(x + self.dropout(new_x))

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn
