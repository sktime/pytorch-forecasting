"""
UniTS: Unified Time Series Model for PyTorch Forecasting.
"""

from typing import Any
import warnings

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers._blocks import _TransformerBlock
from pytorch_forecasting.layers._embeddings import _PatchEmbedding, _PositionalEmbedding
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class UniTS(TslibBaseModel):
    """
    UniTS: Unified Time Series Model.

    GitHub Repository : https://github.com/mims-harvard/UniTS

    Research Paper : https://arxiv.org/abs/2403.00131

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

        # Learnable Task Prompt Tokens
        self.prompt_tokens = nn.Parameter(torch.empty(1, self.prompt_len, self.d_model))
        nn.init.trunc_normal_(self.prompt_tokens, std=0.02)

        self.pos_enc = _PositionalEmbedding(
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
        Forward logic passing data through the abstracted layers.
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
