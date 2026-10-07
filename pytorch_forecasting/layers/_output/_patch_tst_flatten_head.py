"""
Flatten Head for PatchTST model.
"""

import torch.nn as nn


class PatchTSTFlattenHead(nn.Module):
    """
    Flatten Head for the output of the PatchTST model.

    Parameters
    ----------
    patch_num : int
        Number of patches.
    d_model : int
        Model dimension.
    target_window : int
        Target sequence length (prediction_length).
    head_dropout : float, optional
        Dropout rate. Defaults to 0.
    n_quantiles : int, optional
        Number of quantiles. Defaults to 1.
    """

    def __init__(
        self,
        patch_num: int,
        d_model: int,
        target_window: int,
        head_dropout: float = 0.0,
        n_quantiles: int = 1,
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.n_quantiles = n_quantiles

        self.linear = nn.Linear(patch_num * d_model, target_window * n_quantiles)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x: [Batch * n_vars, patch_num, d_model]
        x = self.flatten(x)  # [Batch * n_vars, patch_num * d_model]
        x = self.linear(x)  # [Batch * n_vars, target_window * n_quantiles]
        x = self.dropout(x)

        # return shape: [Batch * n_vars, target_window, n_quantiles]
        batch_vars = x.shape[0]
        x = x.reshape(batch_vars, -1, self.n_quantiles)
        return x
