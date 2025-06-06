"""
Implementation of attention layers from `nn.Module`.
"""

from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Attention layer that combines query, key, and value projections with an attention
    mechanism.
    Args:
        attention (nn.Module): Attention mechanism to use.
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        d_keys (int, optional): Dimension of the keys. Defaults to d_model // n_heads.
        d_values (int, optional):
            Dimension of the values. Defaults to d_model // n_heads.
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        if S == 0:
            # skip the cross attention process since there is no exogenous variables
            queries = self.query_projection(queries)
            return self.out_projection(queries), None

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
