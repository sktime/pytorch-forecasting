"""
Implementation of `nn.Modules` for TimeXer model.
"""

import math
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
        super().__init__()
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


class DataEmbedding_inverted(nn.Module):
    """
    Data embedding module for time series data.
    Args:
        c_in (int): Number of input features.
        d_model (int): Dimension of the model.
        embed_type (str): Type of embedding to use. Defaults to "fixed".
        freq (str): Frequency of the time series data. Defaults to "h".
        dropout (float): Dropout rate. Defaults to 0.1.
    """

    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    """
    Positional embedding module for time series data.
    Args:
        d_model (int): Dimension of the model.
        max_len (int): Maximum length of the input sequence. Defaults to 5000."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class FlattenHead(nn.Module):
    """
    Flatten head for the output of the model.
    Args:
        n_vars (int): Number of input features.
        nf (int): Number of features in the last layer.
        target_window (int): Target window size.
        head_dropout (float): Dropout rate for the head. Defaults to 0.
        n_quantiles (int, optional): Number of quantiles. Defaults to None."""

    def __init__(self, n_vars, nf, target_window, head_dropout=0, n_quantiles=None):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.n_quantiles = n_quantiles

        if self.n_quantiles is not None:
            self.linear = nn.Linear(nf, target_window * n_quantiles)
        else:
            self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)

        if self.n_quantiles is not None:
            batch_size, n_vars = x.shape[0], x.shape[1]
            x = x.reshape(batch_size, n_vars, -1, self.n_quantiles)
        return x


class EnEmbedding(nn.Module):
    """
    Encoder embedding module for time series data. Handles endogenous feature
    embeddings in this case.
    Args:
        n_vars (int): Number of input features.
        d_model (int): Dimension of the model.
        patch_len (int): Length of the patches.
        dropout (float): Dropout rate. Defaults to 0.1."""

    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()

        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    """
    Encoder module for the TimeXer model.
    Args:
        layers (list): List of encoder layers.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.
        projection (nn.Module, optional): Projection layer. Defaults to None.
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    """
    Encoder layer for the TimeXer model.
    Args:
        self_attention (nn.Module): Self-attention mechanism.
        cross_attention (nn.Module): Cross-attention mechanism.
        d_model (int): Dimension of the model.
        d_ff (int, optional):
            Dimension of the feedforward layer. Defaults to 4 * d_model.
        dropout (float): Dropout rate. Defaults to 0.1.
        activation (str): Activation function. Defaults to "relu".
    """

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        )
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(
            self.cross_attention(
                x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
            )[0]
        )
        x_glb_attn = torch.reshape(
            x_glb_attn, (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])
        ).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)
