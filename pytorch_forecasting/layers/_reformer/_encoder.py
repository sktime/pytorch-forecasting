"""Encoder layers for the Reformer-style attention modules.

This module provides a transformer-style encoder layer (`ReformerEncoderLayer`)
that composes an attention module with a feed-forward block implemented via
1D convolutions, layer normalization and dropout. It also provides
`ReformerEncoder`, which stacks multiple encoder layers and optional
convolutional adapters.

These building blocks are used by Reformer-based sequence models to
process token embeddings into contextualized representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReformerEncoderLayer(nn.Module):
    """Single encoder layer combining attention and a conv-based feed-forward.

    Args:
        attention (callable): attention module with signature
            `attention(query, key, value, attn_mask=None, tau=None, delta=None)`
            returning `(output, attn_weights)`.
        d_model (int): model hidden dimensionality.
        d_ff (int or None): intermediate feed-forward dimensionality. If
            `None`, defaults to `4 * d_model`.
        dropout (float): dropout probability.
        activation (str): activation name, currently "relu" or other for
            GELU.
    """

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
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
        """Forward pass through attention and feed-forward blocks.

        Args:
            x (torch.Tensor): input tensor of shape `[batch, seq_len, d_model]`.
            attn_mask (torch.Tensor or None): optional attention mask.
            tau, delta: optional scheduling/hyperparameters forwarded to the
                attention module.

        Returns:
            tuple: `(output, attn)` where `output` has shape
                `[batch, seq_len, d_model]` and `attn` contains attention
                weights or diagnostics from the attention module.
        """
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class ReformerEncoder(nn.Module):
    """Stack multiple `ReformerEncoderLayer`s into a full encoder.

    The encoder optionally interleaves convolutional layers between attention
    layers (used as adapters). The `forward` method returns the final
    encoded tensor and a list of attention diagnostics for each layer.
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """Run the encoder stack.

        Args:
            x (torch.Tensor): input tensor `[batch, seq_len, d_model]`.
            attn_mask (torch.Tensor or None): optional attention mask.
            tau, delta: optional scheduling/hyperparameters forwarded to the
                attention layers.

        Returns:
            tuple: `(x, attns)` where `x` is the encoded output and `attns`
                is a list of attention diagnostics from each layer.
        """
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
