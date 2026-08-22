"""
Patch Embedding Layer for PTF.
"""

import torch
import torch.nn as nn

from pytorch_forecasting.layers._embeddings._positional_embedding import (
    PositionalEmbedding,
)


class UNITS_PatchEmbedding(nn.Module):
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


class PatchEmbedding(nn.Module):
    """
    Patch Embedding module that creates patches and maps them to the model dimension.

    Parameters
    ----------
    d_model : int
        Dimension of the model.
    patch_len : int
        Length of the patch.
    stride : int
        Stride for the patching operation.
    padding : int
        Padding size for the input sequence.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self, d_model: int, patch_len: int, stride: int, padding: int, dropout: float
    ):
        super().__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone: project feature vectors onto d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [Batch * n_vars, 1, seq_len]

        Returns
        -------
        torch.Tensor
            Embedded tensor of shape [Batch * n_vars, patch_num, d_model]
        """
        # x: [Batch * n_vars, 1, seq_len]
        # do padding
        x = self.padding_patch_layer(x)

        # apply patching
        # output shape: [Batch * n_vars, 1, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # reshape for embedding
        # [Batch * n_vars, patch_num, patch_len]
        x = x.squeeze(1)

        # apply embedding
        # [Batch * n_vars, patch_num, d_model]
        x = self.value_embedding(x)

        # add positional embedding
        x = x + self.position_embedding(x)

        return self.dropout(x)
