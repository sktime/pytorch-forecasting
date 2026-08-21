"""
A Time Series is Worth 64 Words (PatchTST)
------------------------------------------
"""

from typing import Any
import warnings as warn

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class PatchTST_v2(BaseModel):
    """
    An implementation of PatchTST model for v2 of pytorch-forecasting.

    PatchTST leverages patching of time series and channel independence to achieve
    state-of-the-art performance on long-term forecasting.

    Parameters
    ----------
    loss: nn.Module
        Loss function to use for training.
    enc_in: int, optional
        Number of input features. If not provided, it is inferred from data.
    hidden_size: int, default=16
        Dimension of the model embeddings.
    n_heads: int, default=2
        Number of attention heads.
    patch_len: int, default=16
        Length of each non-overlapping patch.
    stride: int, default=8
        Stride size for patching.
    padding: int, default=0
        Padding size for the input sequence.
    dropout: float, default=0.1
        Dropout rate.
    head_dropout: float, default=0.1
        Dropout rate for the output head.
    logging_metrics: list[nn.Module] | None, default=None
        List of metrics to log.
    optimizer: Optimizer | str | None, default='adam'
        Optimizer to use for training.
    optimizer_params: dict | None, default=None
        Parameters for the optimizer.
    lr_scheduler: str | None, default=None
        Learning rate scheduler.
    lr_scheduler_params: dict | None, default=None
        Parameters for the scheduler.
    metadata: dict | None, default=None
        Metadata from TslibDataModule.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.patch_tst._patch_tst_pkg_v2 import (
            PatchTST_pkg_v2,
        )

        return PatchTST_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        enc_in: int | None = None,
        hidden_size: int = 16,
        n_heads: int = 2,
        patch_len: int = 16,
        stride: int = 8,
        padding: int = 0,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
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
        )

        self.metadata = metadata or {}

        # Set properties required by the model from metadata
        self.context_length = self.metadata.get("max_encoder_length", 0)
        self.prediction_length = self.metadata.get("max_prediction_length", 0)

        # In BaseModelV2, continuous variable counts are in 'encoder_cont'
        self.cont_dim = self.metadata.get("encoder_cont", 0)

        warn.warn(
            "PatchTST is an experimental model implemented on BaseModelV2. "
            "It is an unstable version and may be subject to unannounced changes."
        )

        self.enc_in = enc_in
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """Initialize the network for PatchTST architecture."""
        from pytorch_forecasting.layers import (
            PatchEmbedding,
            PatchTSTFlattenHead,
        )

        rem = self.context_length % self.patch_len
        self.pad_len = 0 if rem == 0 else self.patch_len - rem

        self.patch_num = int(
            (self.context_length + self.pad_len + self.padding - self.patch_len)
            / self.stride
            + 1
        )

        self.enc_in = self.enc_in or self.cont_dim

        self.n_quantiles = 1
        if hasattr(self.loss, "quantiles") and self.loss.quantiles is not None:
            self.n_quantiles = len(self.loss.quantiles)

        if self.hidden_size % self.n_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )

        self.patch_embedding = PatchEmbedding(
            self.hidden_size, self.patch_len, self.stride, self.padding, self.dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.head = PatchTSTFlattenHead(
            self.patch_num,
            self.hidden_size,
            self.prediction_length,
            head_dropout=self.head_dropout,
            n_quantiles=self.n_quantiles,
        )

    def _forecast(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        """
        batch_size = x.get("encoder_cont", torch.empty(1)).shape[0]

        # Combine endogenous and exogenous variables as continuous covariates
        encoder_target = x.get(
            "encoder_target",
            torch.zeros(batch_size, self.context_length, 1, device=self.device),
        )
        # Handle the case where encoder_target might not have a feature dimension
        if encoder_target.dim() == 2:
            encoder_target = encoder_target.unsqueeze(-1)

        encoder_cont = x.get(
            "encoder_cont",
            torch.empty(batch_size, self.context_length, 0, device=self.device),
        )

        # [Batch, seq_len, total_vars]
        combined = torch.cat([encoder_target, encoder_cont], dim=-1)
        total_vars = combined.shape[-1]

        # Determine target dimension for slicing at the end
        target_dim = encoder_target.shape[-1]

        # Channel Independence: merge Batch and Var dimension
        # [Batch * total_vars, seq_len, 1]
        seq_len = combined.size(1)
        enc_in = combined.permute(0, 2, 1).reshape(batch_size * total_vars, 1, seq_len)

        # Pad sequence if necessary to match the expected patch_num
        total_pad_len = max(0, self.context_length + self.pad_len - seq_len)
        if total_pad_len > 0:
            import torch.nn.functional as F

            enc_in = F.pad(enc_in, (0, total_pad_len))

        # Embedding
        # [Batch * total_vars, patch_num, d_model]
        enc_out = self.patch_embedding(enc_in)

        # Transformer Encoder
        enc_out = self.encoder(enc_out)

        # Flatten Head
        # [Batch * total_vars, target_window, n_quantiles]
        dec_out = self.head(enc_out)

        # Reshape back to separate batch and variables
        # [Batch, total_vars, target_window, n_quantiles]
        dec_out = dec_out.reshape(
            batch_size, total_vars, self.prediction_length, self.n_quantiles
        )

        # Extract only the target predictions
        target_predictions = dec_out[:, :target_dim, :, :]

        # Expected output shape: [Batch, target_window, n_quantiles] if target_dim == 1,
        # or [Batch, target_window, target_dim, n_quantiles] if target_dim > 1.
        # Following TslibBaseModel convention:
        if target_dim == 1:
            target_predictions = target_predictions.squeeze(1)
        else:
            target_predictions = target_predictions.permute(0, 2, 1, 3)

        return target_predictions

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass.
        """
        prediction = self._forecast(x)

        return {"prediction": prediction}
