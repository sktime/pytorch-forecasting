"""
Autoformer decoder with progressive decomposition architecture.

Implements decoder layers that incorporate series decomposition after each
self-attention, cross-attention, and feed-forward sub-layer, progressively
extracting and accumulating trend components.
"""

import torch.nn as nn
import torch.nn.functional as F

from pytorch_forecasting.layers._decomposition._series_decomp import (
    SeriesDecomposition,
)


class AutoformerDecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture.
    """

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        c_out,
        d_ff=None,
        moving_avg=25,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False
        )
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.decomp3 = SeriesDecomposition(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(
            self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
        )
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(
            1, 2
        )
        return x, residual_trend


class AutoformerDecoder(nn.Module):
    """
    Autoformer decoder.
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
