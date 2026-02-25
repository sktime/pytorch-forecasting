"""
Full Attention Layer.
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
        output_attention (bool): Whether to output attention weights.
        use_efficient_attention (bool): Whether to use PyTorch's native,
            optimized Scaled Dot Product Attention implementation which can
            reduce computation time and memory consumption for longer sequences.
            PyTorch automatically selects the optimal backend (FlashAttention-2,
            Memory-Efficient Attention, or their own C++ implementation) based
            on user's input properties, hardware capabilities, and build
            configuration.
    """

    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
        use_efficient_attention=False,
    ):
        super().__init__()

        if output_attention and use_efficient_attention:
            raise ValueError(
                "Cannot output attention scores using efficient attention. "
                "Set `use_efficient_attention=False` or "
                "`output_attention=False`."
            )

        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.use_efficient_attention = use_efficient_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        if self.use_efficient_attention:
            V, A = self._efficient_attention(queries, keys, values, attn_mask)
        else:
            V, A = self._einsum_attention(queries, keys, values, attn_mask)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

    def _einsum_attention(self, queries, keys, values, attn_mask):
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

        return V, A

    def _efficient_attention(self, queries, keys, values, attn_mask):
        # SDPA expects [B, H, L, E] shape
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        V = nn.functional.scaled_dot_product_attention(
            query=queries,
            key=keys,
            value=values,
            attn_mask=attn_mask.mask if attn_mask is not None else None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.mask_flag if attn_mask is None else False,
            scale=self.scale,  # if == None, PyTorch computes internally
        )

        V = V.transpose(1, 2)

        return V, None
