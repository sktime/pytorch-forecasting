"""Temporal embeddings for time features.

This module provides FixedEmbedding, a sinusoidal (fixed) positional
embedding implementation, and TemporalEmbedding, which composes embeddings
for common time features (month, day, weekday, hour, minute).

The embeddings are used to turn temporal indices or time features into
dense vectors that can be added to value/token embeddings in sequence models.
"""

import math

import torch
import torch.nn as nn


class FixedEmbedding(nn.Module):
    """Fixed positional embedding using sinusoidal encodings.

    Args:
        c_in (int): number of discrete positions (e.g., months, hours).
        d_model (int): embedding dimensionality.

    The weights are initialized with sinusoidal encodings and are frozen
    (non-trainable) to provide deterministic, non-learned positional signals.
    """

    def __init__(self, c_in, d_model):
        super().__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """Return the fixed embeddings for the input indices.

        Args:
            x (torch.Tensor): integer tensor of indices (any shape).

        Returns:
            torch.Tensor: embedding vectors with the same leading shape as
                `x` and last dimension `d_model`.
        """
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """Compose embeddings for temporal components.

    Combines embeddings for month, day, weekday, hour and (optionally)
    minute depending on the provided `freq`. Each component uses either
    a fixed sinusoidal embedding or a learned `nn.Embedding` based on
    `embed_type`.

    Args:
        d_model (int): embedding dimensionality.
        embed_type (str): 'fixed' to use FixedEmbedding; otherwise uses
            `nn.Embedding` (learned embeddings).
        freq (str): frequency code; if 't' (minute frequency) a minute
            embedding is included.
    """

    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super().__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        """Return the sum of component temporal embeddings.

        Args:
            x (torch.Tensor): integer tensor of shape
                `[batch, seq_len, num_time_features]` where the last
                dimension contains `[month, day, weekday, hour, minute]`
                (minute optional depending on `freq`).

        Returns:
            torch.Tensor: summed temporal embedding of shape
                `[batch, seq_len, d_model]`.
        """
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x
