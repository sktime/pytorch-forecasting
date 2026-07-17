"""
Patch Embedding Layer for PatchTST.

Adapted from thuml/Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/layers/Embed.py
Paper: https://arxiv.org/pdf/2211.14730.pdf
"""

import torch
import torch.nn as nn

from pytorch_forecasting.layers._embeddings._positional_embedding import (
    PositionalEmbedding,
)


class PatchEmbedding(nn.Module):
    """
    Patch Embedding for time series data.

    Splits each variable's time series into overlapping patches of fixed length,
    then projects each patch into a d_model-dimensional vector space. This is
    the core input representation used by PatchTST.

    The patching operation treats each variable (channel) independently,
    which is referred to as the "Channel Independence" (CI) assumption.

    Parameters
    ----------
    d_model : int
        Dimension of the output embedding for each patch.
    patch_len : int
        Length of each patch (number of time steps per patch).
    stride : int
        Step size between consecutive patches (controls overlap).
    padding : int
        Amount of replication padding applied to the right end of
        the time series before patching. Setting ``padding = stride``
        ensures no time-step information is dropped.
    dropout : float
        Dropout probability applied after the embedding.

    Notes
    -----
    The forward method merges the batch and variable dimensions together
    so that the Transformer encoder can treat each (batch, variable) pair
    as an independent sequence of patch tokens. The returned ``n_vars``
    value is needed by the caller to un-merge these dimensions after
    the encoder has run.
    """

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        stride: int,
        padding: int,
        dropout: float,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        # Pad the right side of the input so no data is lost at the boundary
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Linear projection: each patch of length `patch_len` → vector of size `d_model`
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Sinusoidal positional embedding added on top of patch projections
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Forward pass of the patch embedding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, n_vars, seq_len)``.
            Note: the time dimension is the *last* axis here (already transposed
            from the ``(batch, time, vars)`` format used elsewhere).

        Returns
        -------
        x : torch.Tensor
            Embedded patches of shape
            ``(batch_size * n_vars, n_patches, d_model)``.
        n_vars : int
            Number of input variables (channels). The caller needs this
            to reshape the encoder output back to
            ``(batch_size, n_vars, n_patches, d_model)``.
        """
        # Record n_vars before merging dimensions
        n_vars = x.shape[1]

        # Pad the right end so the unfold produces a clean set of patches
        x = self.padding_patch_layer(x)

        # Slice into overlapping patches: result shape is
        # (batch_size, n_vars, n_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Merge batch and variable dimensions so the Transformer sees each
        # (sample, variable) pair as an independent sequence of patch tokens.
        # Shape becomes (batch_size * n_vars, n_patches, patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        # Project each patch + add positional encoding
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x), n_vars
