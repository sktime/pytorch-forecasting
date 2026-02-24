"""Lightweight wrapper for a Reformer LSH self-attention layer.

This module wraps the `LSHSelfAttention` from the `reformer_pytorch`
package and provides utilities to pad the input length so that it fits
the bucket constraints required by the LSH attention implementation.
"""

import torch
import torch.nn as nn


class ReformerLayer(nn.Module):
    """Wrapper exposing an LSH self-attention as a module compatible with
    the project's attention API.

    Args:
        attention: unused (kept for API compatibility).
        d_model (int): input/output dimensionality.
        n_heads (int): number of attention heads.
        d_keys, d_values: unused placeholders for key/value dims.
        causal (bool): whether attention should be causal.
        bucket_size (int): LSH bucket size, used for padding computation.
        n_hashes (int): number of hash rounds used by LSH attention.
    """

    def __init__(
        self,
        attention,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None,
        causal=False,
        bucket_size=4,
        n_hashes=4,
    ):
        super().__init__()
        from reformer_pytorch import LSHSelfAttention

        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal,
        )

    def fit_length(self, queries):
        """Pad `queries` so its sequence length is divisible by `bucket_size*2`.

        The underlying LSH attention requires the sequence length to be a
        multiple of `bucket_size * 2`. If the length does not meet this
        constraint, zeros are appended to the sequence to satisfy it.

        Args:
            queries (torch.Tensor): tensor `[batch, seq_len, channels]`.

        Returns:
            torch.Tensor: padded tensor with adjusted sequence length.
        """
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat(
                [queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1
            )

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        """Run LSH self-attention on `queries` and return the processed
        tensor.

        The implementation currently ignores `keys`, `values`, `attn_mask`,
        `tau`, and `delta` and returns `(output, None)` where `output` is the
        attention result trimmed back to the original sequence length.

        Args:
            queries (torch.Tensor): input `[batch, seq_len, dim]`.
            keys, values, attn_mask, tau, delta: accepted for API
                compatibility but not used by this wrapper.

        Returns:
            tuple: `(output, None)` where `output` has shape
                `[batch, seq_len, dim]`.
        """
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None
