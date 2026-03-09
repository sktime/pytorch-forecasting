"""
UniTS: Unified Time Series Model for PyTorch Forecasting.
"""

import math
import warnings
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class _PatchEmbedding(nn.Module):
    """Project strided patches of a multivariate time series into d_model space.

    Uses channel-independent patching: each channel's patches are projected
    separately with a shared Linear(patch_len, d_model), then averaged across
    channels. This matches the channel-independent approach in the original paper
    and makes the module robust to any number of input channels at runtime.

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
        return emb.mean(dim=1)


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int
        Maximum sequence length.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        half = d_model // 2
        div_term = torch.exp(
            torch.arange(0, half, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].size(1)])
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1), :])


class _TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block (MHSA + FFN).

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class UniTS(TslibBaseModel):
    """
    UniTS: Unified Time Series Model.

    Patch-based transformer for multivariate time series forecasting.
    Implements a simplified version of the architecture from the paper, using
    channel-independent patching: each channel is projected separately with a
    shared linear layer, then averaged across channels.

    Parameters
    ----------
    loss : nn.Module
        Loss function for training.
    d_model : int, optional
        Transformer model dimension. Default is 64.
    n_heads : int, optional
        Number of self-attention heads. Must divide d_model. Default is 8.
    e_layers : int, optional
        Number of transformer encoder layers. Default is 3.
    d_ff : int, optional
        Feed-forward hidden dimension. Default is 512.
    dropout : float, optional
        Dropout probability. Default is 0.1.
    patch_len : int, optional
        Patch length in time steps. Must be <= context_length. Default is 16.
    stride : int, optional
        Stride between patches. Default is 8.
    prompt_len : int, optional
        Number of learnable task-prompt tokens prepended to patch sequence.
        Default is 10.
    logging_metrics : list[nn.Module] or None, optional
        Metrics to log during training.
    optimizer : Optimizer or str or None, optional
        Optimizer. Default is 'adam'.
    optimizer_params : dict or None, optional
        Optimizer parameters.
    lr_scheduler : str or None, optional
        Learning rate scheduler.
    lr_scheduler_params : dict or None, optional
        Scheduler parameters.
    metadata : dict or None, optional
        Dataset metadata provided by TslibDataModule.

    References
    ----------
    .. [1] Gao, S. et al. (2024). UniTS: Building a Unified Time Series Model.
       https://arxiv.org/abs/2403.00131
    .. [2] https://github.com/mims-harvard/UniTS
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.units._units_pkg_v2 import UniTS_pkg_v2

        return UniTS_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        d_model: int = 64,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        patch_len: int = 16,
        stride: int = 8,
        prompt_len: int = 10,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            metadata=metadata,
        )

        warnings.warn(
            "UniTS is an experimental model implemented on TslibBaseModelV2. "
            "It is an unstable version and may be subject to unannounced changes. "
            "Please use with caution.",
            UserWarning,
            stacklevel=2,
        )

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )

        if patch_len > self.context_length:
            raise ValueError(
                f"patch_len ({patch_len}) must not exceed "
                f"context_length ({self.context_length})."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.patch_len = patch_len
        self.stride = stride
        self.prompt_len = prompt_len

        self._init_network()

    def _init_network(self):
        """Initialise model layers."""
        self.num_patches = max(
            1, (self.context_length - self.patch_len) // self.stride + 1
        )

        self.patch_embedding = _PatchEmbedding(
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            dropout=self.dropout,
        )

        self.prompt_tokens = nn.Parameter(torch.empty(1, self.prompt_len, self.d_model))
        nn.init.trunc_normal_(self.prompt_tokens, std=0.02)

        self.pos_enc = _PositionalEncoding(
            d_model=self.d_model,
            max_len=self.prompt_len + self.num_patches + 16,
            dropout=self.dropout,
        )

        self.encoder = nn.ModuleList(
            [
                _TransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout)
                for _ in range(self.e_layers)
            ]
        )

        self.norm = nn.LayerNorm(self.d_model)

        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                self.num_patches * self.d_model,
                self.prediction_length * self.target_dim,
            ),
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input batch. Uses ``history_target`` (batch, context_length, target_dim)
            and optionally ``history_cont`` (batch, context_length, cont_dim).

        Returns
        -------
        dict[str, torch.Tensor]
            ``prediction`` of shape (batch, prediction_length, target_dim).
        """
        target = x["history_target"]
        B = target.size(0)

        cont = x.get("history_cont")
        if cont is not None and cont.size(-1) > 0:
            src = torch.cat([cont, target], dim=-1)
        else:
            src = target

        mean = src.mean(dim=1, keepdim=True)
        std = src.std(dim=1, keepdim=True, unbiased=False) + 1e-5
        src = (src - mean) / std

        patch_emb = self.patch_embedding(src)
        seq = torch.cat([self.prompt_tokens.expand(B, -1, -1), patch_emb], dim=1)
        seq = self.pos_enc(seq)

        for layer in self.encoder:
            seq = layer(seq)
        seq = self.norm(seq)

        patch_out = seq[:, self.prompt_len : self.prompt_len + self.num_patches, :]
        out = self.head(patch_out).view(B, self.prediction_length, self.target_dim)

        target_mean = mean[:, :, -self.target_dim :]
        target_std = std[:, :, -self.target_dim :]
        return {"prediction": out * target_std + target_mean}
