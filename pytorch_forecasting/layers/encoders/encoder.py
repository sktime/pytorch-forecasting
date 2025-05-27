"""
Implementation of encoder layers from `nn.Module`.
"""

import math
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder module for the TimeXer model.
    Args:
        layers (list): List of encoder layers.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.
        projection (nn.Module, optional): Projection layer. Defaults to None.
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
