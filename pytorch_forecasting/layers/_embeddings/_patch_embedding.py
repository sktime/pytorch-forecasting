"""
Patch Embedding Layer for PTF.
"""

import torch
import torch.nn as nn

from pytorch_forecasting.layers._embeddings._positional_embedding import (
    PositionalEmbedding,
)


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
