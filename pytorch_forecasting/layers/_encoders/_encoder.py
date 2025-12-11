"""
Implementation of encoder layers from `nn.Module`.
"""

import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder module for Tslib models.
    Args:
        layers (list): List of encoder layers.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.
        projection (nn.Module, optional): Projection layer. Defaults to None.
        output_attention (Boolean, optional): Whether to output attention weights. Defaults to False.
    """  # noqa: E501

    def __init__(
        self, layers, norm_layer=None, projection=None, output_attention=False
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        self.output_attention = output_attention

    def forward(
        self, x, cross=None, x_mask=None, cross_mask=None, tau=None, delta=None
    ):
        if self.output_attention:
            attns = []
            for layer in self.layers:
                x, attn = layer(
                    x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
                )
                attns.append(attn)
        else:
            for layer in self.layers:
                x = layer(
                    x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
                )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        if self.output_attention:
            return x, attns
        return x
