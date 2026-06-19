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

    Parameters
    ----------
    layers : list
        List of encoder layers.
    norm_layer : nn.Module, optional
        Normalization layer. Default is None.
    projection : nn.Module, optional
        Projection layer. Default is None.

    Attributes
    ----------
    layers : nn.ModuleList
        Module list containing the encoder layers.
    norm : nn.Module or None
        Normalization layer instance.
    projection : nn.Module or None
        Projection layer instance.

    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_model).
        cross : torch.Tensor
            Cross-attention input tensor of shape (batch_size, seq_len_cross, d_model).
        x_mask : torch.Tensor, optional
            Attention mask for self-attention. Default is None.
        cross_mask : torch.Tensor, optional
            Attention mask for cross-attention. Default is None.
        tau : torch.Tensor, optional
            Temporal parameter for attention mechanisms. Default is None.
        delta : torch.Tensor, optional
            Delta parameter for cross-attention. Default is None.

        Returns
        -------
        torch.Tensor
            Encoded output tensor.
        """
        for layer in self.layers:
            x = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
