"""
Implementation of attention layers from `nn.Module`.
"""

from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriangularCausalMask:
    """
    Triangular causal mask for attention mechanism.
    """

    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    """
    Full attention mechanism with optional masking and dropout.
    Args:
        mask_flag (bool): Whether to apply masking.
        factor (int): Factor for scaling the attention scores.
        scale (float): Scaling factor for attention scores.
        attention_dropout (float): Dropout rate for attention scores.
        output_attention (bool): Whether to output attention weights."""

    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.abs)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


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
        super(AttentionLayer, self).__init__()

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
