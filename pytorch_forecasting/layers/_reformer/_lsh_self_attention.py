"""
Self-contained LSH Self-Attention implementation.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _look_one_back(x: torch.Tensor) -> torch.Tensor:
    """Concatenate each bucket with its left neighbour."""
    x_extra = torch.cat([x[:, -1:], x[:, :-1]], dim=1)
    return torch.cat([x_extra, x], dim=2)


class LSHAttention(nn.Module):
    """Low-level LSH attention operating on already-projected QK and V.

    Args:
        bucket_size (int): tokens per bucket half (full bucket = 2*bucket_size).
        n_hashes (int): number of hash rounds; outputs are averaged.
        causal (bool): mask future tokens.
        allow_duplicate_attention (bool): let a token attend to itself twice
            (once as Q, once as K).
        attend_across_buckets (bool): each bucket also attends to the
            preceding bucket (doubles the attended set).
        drop_for_hash_rate (float): dropout on hash vectors.
        dropout (float): dropout on attention weights.
    """

    def __init__(
        self,
        bucket_size: int = 64,
        n_hashes: int = 8,
        causal: bool = False,
        allow_duplicate_attention: bool = True,
        attend_across_buckets: bool = True,
        drop_for_hash_rate: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.causal = causal
        self.allow_duplicate_attention = allow_duplicate_attention
        self.attend_across_buckets = attend_across_buckets
        self.hash_dropout = nn.Dropout(p=drop_for_hash_rate)
        self.attn_dropout = nn.Dropout(p=dropout)

    def _hash_vectors(
        self, vecs: torch.Tensor, n_buckets: int, random_rotations: torch.Tensor
    ) -> torch.Tensor:
        """Assign each vector to one of `n_buckets` buckets via random projections.

        Args:
            vecs: (batch, seq_len, dim_head)
            n_buckets: total number of buckets
            random_rotations: (dim_head, n_hashes, n_buckets // 2)

        Returns:
            buckets: (batch, n_hashes * seq_len)  integer bucket indices
        """
        batch, _, _ = vecs.shape
        vecs = self.hash_dropout(vecs)

        rotated = torch.einsum("bld,dhr->blhr", vecs, random_rotations)

        rotated = torch.cat([rotated, -rotated], dim=-1)
        buckets = rotated.argmax(dim=-1)

        buckets = buckets.permute(0, 2, 1).reshape(batch, -1)
        return buckets

    def forward(
        self,
        qk: torch.Tensor,
        v: torch.Tensor,
        input_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            qk: (batch, seq_len, dim_head)  — shared Q=K projection
            v:  (batch, seq_len, dim_head)
            input_mask: (batch, seq_len) bool, True = keep

        Returns:
            out: (batch, seq_len, dim_head)
        """
        batch, seq_len, dim = qk.shape
        bucket_size = self.bucket_size
        n_hashes = self.n_hashes

        assert seq_len % (bucket_size * 2) == 0, (
            f"Sequence length {seq_len} must be divisible by \
            bucket_size*2={bucket_size*2}. "
            "Use ReformerLayer.fit_length() to pad first."
        )

        n_buckets = seq_len // bucket_size

        random_rotations = torch.randn(
            dim, n_hashes, n_buckets // 2, device=qk.device, dtype=qk.dtype
        )
        qk_norm = F.normalize(qk, p=2, dim=-1)
        buckets = self._hash_vectors(qk_norm, n_buckets, random_rotations)

        ticker = torch.arange(n_hashes * seq_len, device=qk.device).unsqueeze(0)
        buckets_and_t = seq_len * buckets + (ticker % seq_len)

        _, sorted_idx = buckets_and_t.sort(dim=-1)
        _, unsorted_idx = sorted_idx.sort(dim=-1)

        qk_tiled = (
            qk_norm.unsqueeze(1).expand(-1, n_hashes, -1, -1).reshape(batch, -1, dim)
        )
        v_tiled = v.unsqueeze(1).expand(-1, n_hashes, -1, -1).reshape(batch, -1, dim)

        exp_idx = sorted_idx.unsqueeze(-1).expand(-1, -1, dim)
        qk_sorted = qk_tiled.gather(1, exp_idx)
        v_sorted = v_tiled.gather(1, exp_idx)

        chunk_size = bucket_size * 2
        total_len = n_hashes * seq_len

        n_chunks = total_len // chunk_size
        qk_chunks = qk_sorted.reshape(batch, n_chunks, chunk_size, dim)
        v_chunks = v_sorted.reshape(batch, n_chunks, chunk_size, dim)

        if self.attend_across_buckets:
            qk_attend = _look_one_back(qk_chunks)
            v_attend = _look_one_back(v_chunks)
        else:
            qk_attend = qk_chunks
            v_attend = v_chunks

        scale = dim**-0.5
        dots = torch.einsum("bcqd,bckd->bcqk", qk_chunks * scale, qk_attend)
        sorted_idx_chunks = sorted_idx.reshape(batch, n_chunks, chunk_size)
        true_pos_q = sorted_idx_chunks % seq_len

        if self.attend_across_buckets:
            true_pos_k_extra = torch.cat(
                [true_pos_q[:, -1:], true_pos_q[:, :-1]], dim=1
            )
            true_pos_k = torch.cat([true_pos_k_extra, true_pos_q], dim=2)
        else:
            true_pos_k = true_pos_q

        if not self.allow_duplicate_attention:
            dupe_mask = true_pos_q.unsqueeze(-1) == true_pos_k.unsqueeze(-2)
            dots.masked_fill_(dupe_mask, float("-inf"))

        if self.causal:
            causal_mask = true_pos_q.unsqueeze(-1) < true_pos_k.unsqueeze(-2)
            dots.masked_fill_(causal_mask, float("-inf"))

        if input_mask is not None:
            mask_tiled = (
                input_mask.unsqueeze(1).expand(-1, n_hashes, -1).reshape(batch, -1)
            )
            mask_sorted_idx = sorted_idx.reshape(batch, n_chunks, chunk_size)
            if self.attend_across_buckets:
                k_extra_idx = torch.cat(
                    [mask_sorted_idx[:, -1:], mask_sorted_idx[:, :-1]], dim=1
                )
                k_idx = torch.cat([k_extra_idx, mask_sorted_idx], dim=2)
            else:
                k_idx = mask_sorted_idx
            orig_k_pos = k_idx % seq_len
            k_mask = mask_tiled.gather(1, orig_k_pos.reshape(batch, -1)).reshape(
                batch, n_chunks, -1
            )
            dots.masked_fill_(~k_mask.unsqueeze(2).bool(), float("-inf"))

        attn = F.softmax(dots, dim=-1)
        attn = self.attn_dropout(attn)

        out_chunks = torch.einsum("bcqk,bckd->bcqd", attn, v_attend)

        out_sorted = out_chunks.reshape(batch, total_len, dim)

        exp_unsort_idx = unsorted_idx.unsqueeze(-1).expand(-1, -1, dim)
        out_unsorted = out_sorted.gather(1, exp_unsort_idx)

        out = out_unsorted.reshape(batch, n_hashes, seq_len, dim).mean(dim=1)

        return out


class LSHSelfAttention(nn.Module):
    """Multi-head LSH Self-Attention layer.

    Args:
        dim (int): model dimensionality (input and output).
        heads (int): number of attention heads.
        bucket_size (int): tokens per half-bucket; sequence length must be
            divisible by ``bucket_size * 2``.
        n_hashes (int): number of LSH hash rounds to average over.
        causal (bool): enable causal (autoregressive) masking.
        dim_head (int | None): dimensionality per head; defaults to
            ``dim // heads``.
        attn_chunks (int): process attention in this many sequential chunks
            to trade speed for memory (1 = no chunking).
        dropout (float): dropout on attention weights.
        post_attn_dropout (float): dropout after the output projection.
        allow_duplicate_attention (bool): passed to ``LSHAttention``.
        attend_across_buckets (bool): passed to ``LSHAttention``.
        num_mem_kv (int): number of persistent memory key-value pairs appended
            to every sequence (similar to "all-attention" paper).
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        bucket_size: int = 64,
        n_hashes: int = 8,
        causal: bool = False,
        dim_head: int | None = None,
        attn_chunks: int = 1,
        dropout: float = 0.0,
        post_attn_dropout: float = 0.0,
        allow_duplicate_attention: bool = True,
        attend_across_buckets: bool = True,
        num_mem_kv: int = 0,
    ):
        super().__init__()
        dim_head = dim_head or (dim // heads)
        inner_dim = dim_head * heads

        self.heads = heads
        self.dim_head = dim_head
        self.bucket_size = bucket_size
        self.attn_chunks = attn_chunks
        self.num_mem_kv = num_mem_kv

        self.to_qk = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(post_attn_dropout),
        )

        self.lsh_attn = LSHAttention(
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal,
            allow_duplicate_attention=allow_duplicate_attention,
            attend_across_buckets=attend_across_buckets,
            dropout=dropout,
        )

        if num_mem_kv > 0:
            self.mem_kv = nn.Parameter(torch.randn(1, num_mem_kv, dim_head * 2))
        else:
            self.mem_kv = None

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            input_mask: (batch, seq_len) bool tensor, True = valid token.

        Returns:
            (batch, seq_len, dim)
        """
        batch, seq_len, _ = x.shape
        heads = self.heads
        dim_h = self.dim_head

        qk = self.to_qk(x)
        v = self.to_v(x)

        def split_heads(t):
            return t.reshape(batch, seq_len, heads, dim_h).permute(0, 2, 1, 3)

        qk = split_heads(qk)
        v = split_heads(v)

        if self.mem_kv is not None:
            mem = self.mem_kv.expand(batch, -1, -1)
            mem_k, mem_v = mem.chunk(2, dim=-1)
            mem_k = mem_k.unsqueeze(1).expand(-1, heads, -1, -1)
            mem_v = mem_v.unsqueeze(1).expand(-1, heads, -1, -1)
            qk = torch.cat([qk, mem_k], dim=2)
            v = torch.cat([v, mem_v], dim=2)
            if input_mask is not None:
                mem_mask = input_mask.new_ones(batch, self.num_mem_kv)
                input_mask = torch.cat([input_mask, mem_mask], dim=1)

        total_seq = qk.shape[2]
        qk_flat = qk.reshape(batch * heads, total_seq, dim_h)
        v_flat = v.reshape(batch * heads, total_seq, dim_h)

        if input_mask is not None:
            mask_flat = (
                input_mask.unsqueeze(1).expand(-1, heads, -1).reshape(batch * heads, -1)
            )
        else:
            mask_flat = None

        chunk_size = math.ceil(batch * heads / self.attn_chunks)
        out_chunks = []
        for i in range(0, batch * heads, chunk_size):
            sl = slice(i, i + chunk_size)
            out_chunk = self.lsh_attn(
                qk_flat[sl],
                v_flat[sl],
                input_mask=mask_flat[sl] if mask_flat is not None else None,
            )
            out_chunks.append(out_chunk)

        out = torch.cat(out_chunks, dim=0)

        out = out[:, :seq_len, :]

        out = out.reshape(batch, heads, seq_len, dim_h)
        out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, heads * dim_h)

        return self.to_out(out)
