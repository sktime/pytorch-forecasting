"""
Patch Embedding Layer for PTF.
"""

import torch
import torch.nn as nn


class _PatchEmbedding(nn.Module):
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
