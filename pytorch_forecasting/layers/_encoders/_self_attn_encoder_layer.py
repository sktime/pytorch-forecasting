"""
Self-attention Encoder Layer for PatchTST and similar encoder-only models.

Adapted from thuml/Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/layers/Transformer_EncDec.py
"""

import torch.nn as nn
import torch.nn.functional as F


class SelfAttnEncoderLayer(nn.Module):
    """
    A single Transformer encoder layer using self-attention only.

    Unlike the ``EncoderLayer`` in ``_encoder_layer.py`` (which uses
    both self- and cross-attention for TimeXer), this layer is
    a standard encoder block with:
    - Multi-head self-attention
    - Position-wise feed-forward network (implemented via two 1-D convolutions)
    - Layer normalisation and dropout in both sub-layers (Pre-Net residuals)

    Parameters
    ----------
    attention : nn.Module
        An ``AttentionLayer`` wrapping the inner attention mechanism
        (typically ``FullAttention``).
    d_model : int
        Dimension of the model (embedding size).
    d_ff : int, optional
        Hidden dimension of the feed-forward network.
        Defaults to ``4 * d_model``.
    dropout : float
        Dropout probability in attention and feed-forward sub-layers.
        Defaults to 0.1.
    activation : str
        Activation function for the feed-forward network.
        Must be ``"relu"`` or ``"gelu"``. Defaults to ``"relu"``.
    """

    def __init__(
        self,
        attention,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "relu",
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

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        Forward pass of the encoder layer.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch_size, seq_len, d_model)``.
        attn_mask : optional
            Attention mask (usually ``None`` for PatchTST).

        Returns
        -------
        x : torch.Tensor
            Output of shape ``(batch_size, seq_len, d_model)``.
        attn : torch.Tensor or None
            Attention weights (returned only if ``output_attention=True``
            in the inner ``FullAttention``; otherwise ``None``).
        """
        # --- Self-attention sub-layer ---
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        # --- Feed-forward sub-layer (via 1-D conv, equivalent to linear) ---
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn
