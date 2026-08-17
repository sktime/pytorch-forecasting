import torch
import torch.nn as nn

from pytorch_forecasting.layers._attention._auto_correlation import AutoCorrelation
from pytorch_forecasting.layers._decomposition._autoformer_decomposition import (
    AutoformerDecomposition,
)


class EncoderLayer(nn.Module):
    """
    Single encoder layer used in the Autoformer architecture.

    The layer performs:

    1. Auto-Correlation based attention
    2. Residual connection with layer normalization
    3. Series decomposition into seasonal and trend components

    Parameters
    ----------
    d_model : int
        Hidden dimension of the model.
    moving_avg : int
        Window size used for the decomposition moving average filter.
    """

    def __init__(self, d_model, moving_avg):
        super().__init__()
        self.attn = AutoCorrelation(d_model)
        self.decomp = AutoformerDecomposition(moving_avg)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        new_x, _ = self.attn(x, x, x)
        x = self.norm(x + new_x)
        seasonal, trend = self.decomp(x)
        return seasonal, trend


class Encoder(nn.Module):
    """
    Autoformer encoder composed of stacked encoder layers.

    Each layer progressively decomposes the input sequence into
    seasonal and trend components while modeling long-range
    dependencies using Auto-Correlation.

    Parameters
    ----------
    d_model : int
        Hidden dimension of the model.
    num_layers : int
        Number of encoder layers.
    moving_avg : int
        Window size used in the decomposition block.
    """

    def __init__(self, d_model, num_layers, moving_avg):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, moving_avg) for _ in range(num_layers)]
        )

    def forward(self, x):
        trend = 0
        for layer in self.layers:
            x, t = layer(x)
            trend = trend + t
        return x, trend


class DecoderLayer(nn.Module):
    """
    Decoder layer used in the Autoformer architecture.

    The layer performs cross Auto-Correlation attention between the
    decoder input and encoder outputs, followed by a decomposition
    step to extract seasonal and trend components.

    Parameters
    ----------
    d_model : int
        Hidden dimension of the model.
    moving_avg : int
        Window size used for the decomposition moving average filter.
    """

    def __init__(self, d_model, moving_avg):
        super().__init__()
        self.cross_attn = AutoCorrelation(d_model)
        self.decomp = AutoformerDecomposition(moving_avg)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, cross):
        new_x, _ = self.cross_attn(x, cross, cross)
        x = self.norm(x + new_x)
        seasonal, trend = self.decomp(x)
        return seasonal, trend


class Decoder(nn.Module):
    """
    Autoformer decoder composed of stacked decoder layers.

    The decoder reconstructs future sequences by combining
    encoder representations with seasonal and trend components
    generated through progressive decomposition.

    Parameters
    ----------
    d_model : int
        Hidden dimension of the model.
    num_layers : int
        Number of decoder layers.
    moving_avg : int
        Window size used in the decomposition block.
    """

    def __init__(self, d_model, num_layers, moving_avg):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, moving_avg) for _ in range(num_layers)]
        )

    def forward(self, x, cross):
        trend = 0
        for layer in self.layers:
            x, t = layer(x, cross)
            trend = trend + t
        return x, trend
