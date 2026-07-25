"""
Implementation of `nn.Modules` for PatchTST model.
"""

import math

import torch
import torch.nn as nn


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class PositionalEmbedding(nn.Module):
    """
    Positional embedding for PatchTST.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [Batch, Sequence, d_model]
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    """
    Patch Embedding module that creates patches and maps them to the model dimension.
    """

    def __init__(self, d_model, patch_len, stride, padding, dropout):
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

    def forward(self, x):
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


class FlattenHead(nn.Module):
    """
    Flatten Head for the output of the model.
    """

    def __init__(
        self, patch_num, d_model, target_window, head_dropout=0, n_quantiles=1
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
