"""
PAttn model for PyTorch Forecasting v2.
---------------------------------------
"""

#################################################
# NOTE: This is an experimental implementation  #
# of PAttn for PyTorch Forecasting v2.          #
# It is an unstable API and subject to change.  #
#################################################

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers import (
    AttentionLayer,
    FullAttention,
    PAttnEncoderLayer,
)
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class PAttn(TslibBaseModel):
    """
    PAttn: Patching and Attention for Time Series Forecasting.

    PAttn applies instance normalization, extracts overlapping patches with
    replication padding, embeds patches linearly, processes patch tokens per
    channel with self-attention, and projects to the prediction horizon.

    Parameters
    ----------
    loss : nn.Module
        Loss function for training step.
    d_model : int, default=64
        Hidden dimension of the patch embeddings and attention layers.
    n_heads : int, default=8
        Number of attention heads. Must divide ``d_model``.
    e_layers : int, default=1
        Number of stacked self-attention encoder layers.
    d_ff : int, default=256
        Hidden dimension of the feed-forward network inside each encoder layer.
    patch_len : int, default=16
        Length of each time series patch.
    stride : int, default=8
        Stride between consecutive patches.
    dropout : float, default=0.1
        Dropout probability used in attention and encoder layers.
    factor : int, default=5
        Attention factor passed to ``FullAttention``.
    activation : str, default="gelu"
        Activation function used in the encoder feed-forward network
        (``"relu"`` or ``"gelu"``).
    logging_metrics : list[nn.Module] | None, default=None
        List of metrics to log during training, validation, and testing.
    optimizer : Optimizer | str | None, default="adam"
        Optimizer used for training.
    optimizer_params : dict | None, default=None
        Parameters passed to the optimizer.
    lr_scheduler : str | None, default=None
        Learning rate scheduler name.
    lr_scheduler_params : dict | None, default=None
        Parameters passed to the learning rate scheduler.
    metadata : dict | None, default=None
        Metadata for the model from ``TslibDataModule``.

    References
    ----------
    [1] Are Language Models Actually Useful for Time Series Forecasting?
        (https://arxiv.org/abs/2406.16964).
    [2] https://github.com/thuml/Time-Series-Library/blob/main/models/PAttn.py
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.pattn._pattn_forecaster_v2 import (
            PAttnForecaster,
        )

        return PAttnForecaster

    def __init__(
        self,
        loss: nn.Module,
        d_model: int = 64,
        n_heads: int = 8,
        e_layers: int = 1,
        d_ff: int = 256,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        factor: int = 5,
        activation: str = "gelu",
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

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        if patch_len <= 0:
            raise ValueError(f"patch_len must be positive, got {patch_len}.")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}.")
        if e_layers <= 0:
            raise ValueError(f"e_layers must be positive, got {e_layers}.")
        if activation not in ("relu", "gelu"):
            raise ValueError(
                f"activation must be 'relu' or 'gelu', got {activation!r}."
            )
        if self.context_length > 0 and self.context_length + stride < patch_len:
            raise ValueError(
                f"context_length + stride ({self.context_length + stride}) must be "
                f"at least patch_len ({patch_len})."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.patch_len = patch_len
        self.stride = stride
        self.dropout = dropout
        self.factor = factor
        self.activation = activation

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """Initialize the PAttn network components."""
        self.patch_num = (
            (self.context_length - self.patch_len) // self.stride + 2
            if self.context_length > 0
            else 0
        )

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.in_layer = nn.Linear(self.patch_len, self.d_model)
        self.encoder_layers = nn.ModuleList(
            [
                PAttnEncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        self.d_model,
                        self.n_heads,
                    ),
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.e_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(self.d_model)

        self.n_quantiles = None
        output_dim = self.prediction_length

        if isinstance(self._loss, QuantileLoss):
            self.n_quantiles = len(self._loss.quantiles)
            output_dim *= self.n_quantiles

        self.out_layer = nn.Linear(self.d_model * self.patch_num, output_dim)

    def _encoder(
        self, x: torch.Tensor, target_indices: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Encode input time series through patching and self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, context_length, n_features)``.
        target_indices : torch.Tensor or None
            Indices of target features to extract from the output.
            If None, all features are returned.

        Returns
        -------
        torch.Tensor
            Forecast tensor of shape ``(batch_size, prediction_length, n_targets)``
            for point forecasts, or ``(batch_size, prediction_length, n_quantiles)``
            for quantile forecasts.
        """
        means = x.mean(1, keepdim=True).detach()
        x_norm = x - means
        stdev = torch.sqrt(
            torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_norm = x_norm / stdev

        batch_size, _, n_vars = x_norm.shape
        x_enc = x_norm.permute(0, 2, 1)
        x_enc = self.padding_patch_layer(x_enc)
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        enc_out = self.in_layer(x_enc)
        enc_out = enc_out.reshape(batch_size * n_vars, self.patch_num, self.d_model)
        for layer in self.encoder_layers:
            enc_out, _ = layer(enc_out)
        enc_out = self.encoder_norm(enc_out)

        dec_out = enc_out.reshape(batch_size, n_vars, self.patch_num * self.d_model)
        dec_out = self.out_layer(dec_out)

        if target_indices is not None:
            dec_out = dec_out[:, target_indices, :]
            target_means = means[:, :, target_indices]
            target_stdev = stdev[:, :, target_indices]
        else:
            target_means = means
            target_stdev = stdev

        if self.n_quantiles is None:
            return dec_out.permute(0, 2, 1) * target_stdev + target_means

        if dec_out.shape[1] != 1:
            raise ValueError(
                "Quantile forecasting currently only supports a single target."
            )

        dec_out = dec_out.reshape(batch_size, self.prediction_length, self.n_quantiles)
        return dec_out * target_stdev + target_means

    def _prepare_input_data(
        self, x: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare input data and target indices for model input."""
        available_features = []
        current_idx = 0

        if "history_cont" in x and x["history_cont"].size(-1) > 0:
            available_features.append(x["history_cont"])
            current_idx += x["history_cont"].size(-1)

        if "history_target" not in x or x["history_target"].size(-1) == 0:
            raise ValueError("No target history found in the input dictionary.")

        history_target = x["history_target"]
        if history_target.ndim == 2:
            history_target = history_target.unsqueeze(-1)

        n_targets = history_target.size(-1)
        target_indices = list(range(current_idx, current_idx + n_targets))
        available_features.append(history_target)

        input_data = torch.cat(available_features, dim=-1)
        target_indices_tensor = torch.tensor(
            target_indices, dtype=torch.long, device=input_data.device
        )
        return input_data, target_indices_tensor

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the PAttn model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing input tensors.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the model prediction.
        """
        input_data, target_indices = self._prepare_input_data(x)
        prediction = self._encoder(input_data, target_indices)

        if "target_scale" in x and hasattr(self, "transform_output"):
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
