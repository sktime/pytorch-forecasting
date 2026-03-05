"""
LightTS model for PyTorch Forecasting v2.
----------------------------------------
"""

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers._blocks import IEBlock
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class LightTS(TslibBaseModel):
    """
    LightTS model for long-term time-series forecasting.

    Implements the LightTS architecture with two sampling branches,
    information-exchange blocks, and an autoregressive highway layer.

    References
    ----------
    Paper:
        LightTS: Lightweight Time Series Forecasting with Sampling
        https://arxiv.org/abs/2207.01186

    Original implementation:
        https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py
    """

    @classmethod
    def _pkg(cls):
        """
        Return the package container class associated with LightTS.
        """
        from pytorch_forecasting.models.lightts._lightts_pkg_v2 import LightTS_pkg_v2

        return LightTS_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        d_model: int = 256,
        chunk_size: int = 24,
        dropout: float = 0.1,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the LightTS forecasting model
        Parameters
        ----------
        loss : nn.Module
            Loss function used for training.
        d_model : int, default=256
            Base hidden dimension used to derive internal projection sizes.
        chunk_size : int, default=24
            Chunk size used by the sampling branches.
        dropout : float, default=0.1
            Dropout hyperparameter stored with model configuration.
        logging_metrics : list[nn.Module] or None, default=None
            Additional metrics logged during training/validation/testing.
        optimizer : Optimizer or str or None, default="adam"
            Optimizer instance or optimizer name.
        optimizer_params : dict or None, default=None
            Parameters passed to the optimizer.
        lr_scheduler : str or None, default=None
            Learning-rate scheduler name.
        lr_scheduler_params : dict or None, default=None
            Parameters passed to the learning-rate scheduler.
        metadata : dict or None, default=None
            Metadata from the datamodule (feature dimensions and sequence lengths).
        **kwargs : Any
            Extra keyword arguments accepted for API compatibility.
        """
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            metadata=metadata,
        )

        self.d_model = d_model
        self.chunk_size = chunk_size
        self.dropout = dropout
        self.n_quantiles: int | None = None

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self) -> None:
        """
        Build LightTS layers and projections based on metadata and chunk settings.
        """
        if self.context_length <= 0:
            raise ValueError("context_length must be positive in metadata for LightTS.")
        if self.prediction_length <= 0:
            raise ValueError(
                "prediction_length must be positive in metadata for LightTS."
            )
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        self.enc_in = self.cont_dim + self.target_dim

        # Keep chunk size compatible with both context and horizon.
        self.chunk_size = int(
            min(self.chunk_size, self.context_length, self.prediction_length)
        )
        self.padded_context_length = self.context_length
        if self.padded_context_length % self.chunk_size != 0:
            self.padded_context_length += (
                self.chunk_size - self.padded_context_length % self.chunk_size
            )
        self.num_chunks = self.padded_context_length // self.chunk_size

        if isinstance(self.loss, QuantileLoss):
            self.n_quantiles = len(self.loss.quantiles)
            if self.target_dim != 1:
                raise ValueError(
                    "LightTS currently supports QuantileLoss only for single-target "
                    "forecasting in v2."
                )

        output_dim = self.prediction_length
        if self.n_quantiles is not None:
            output_dim = self.prediction_length * self.n_quantiles

        hidden_dim = max(4, self.d_model // 4)
        merged_dim = hidden_dim * 2

        self.layer_1 = IEBlock(
            input_dim=self.chunk_size,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_nodes=self.num_chunks,
        )
        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)

        self.layer_2 = IEBlock(
            input_dim=self.chunk_size,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_nodes=self.num_chunks,
        )
        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)

        self.layer_3 = IEBlock(
            input_dim=merged_dim,
            hidden_dim=merged_dim,
            output_dim=output_dim,
            num_nodes=self.enc_in,
        )

        self.ar = nn.Linear(self.padded_context_length, output_dim)

    def _prepare_input_data(
        self, x: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Prepare model inputs from the batch dictionary.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input batch dictionary containing historical feature tensors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Concatenated input tensor and optional target-channel indices.
        """

        available_features = []
        target_indices = []
        current_idx = 0

        if "history_cont" in x and x["history_cont"].size(-1) > 0:
            available_features.append(x["history_cont"])
            current_idx += x["history_cont"].size(-1)

        if "history_target" in x and x["history_target"].size(-1) > 0:
            n_targets = x["history_target"].size(-1)
            target_indices = list(range(current_idx, current_idx + n_targets))
            available_features.append(x["history_target"])

        if not available_features:
            raise ValueError("No valid input features found in the input dictionary.")

        input_data = torch.cat(available_features, dim=-1)
        target_idx_tensor = None
        if target_indices:
            target_idx_tensor = torch.tensor(
                target_indices, dtype=torch.long, device=input_data.device
            )

        return input_data, target_idx_tensor

    def _encoder(
        self, x: torch.Tensor, target_indices: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Run the LightTS encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, context_length, features).
        target_indices : torch.Tensor or None
            Indices of target channels.

        Returns
        -------
        torch.Tensor
            Forecast tensor.
        """
        batch_size, seq_len, n_features = x.shape

        if seq_len > self.padded_context_length:
            x = x[:, -self.padded_context_length :, :]
            seq_len = self.padded_context_length

        if seq_len < self.padded_context_length:
            pad_len = self.padded_context_length - seq_len
            x = torch.cat(
                [
                    x,
                    torch.zeros(
                        batch_size, pad_len, n_features, dtype=x.dtype, device=x.device
                    ),
                ],
                dim=1,
            )

        # Highway (autoregressive) branch.
        highway = self.ar(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Continuous sampling.
        x1 = x.reshape(batch_size, self.num_chunks, self.chunk_size, n_features)
        x1 = x1.permute(0, 3, 2, 1).reshape(-1, self.chunk_size, self.num_chunks)
        x1 = self.layer_1(x1)
        x1 = self.chunk_proj_1(x1).squeeze(dim=-1)

        # Interval sampling.
        x2 = x.reshape(batch_size, self.chunk_size, self.num_chunks, n_features)
        x2 = x2.permute(0, 3, 1, 2).reshape(-1, self.chunk_size, self.num_chunks)
        x2 = self.layer_2(x2)
        x2 = self.chunk_proj_2(x2).squeeze(dim=-1)

        x3 = torch.cat([x1, x2], dim=-1)
        x3 = x3.reshape(batch_size, n_features, -1).permute(0, 2, 1)

        output = self.layer_3(x3) + highway

        if target_indices is not None:
            output = output[:, :, target_indices]

        return self._reshape_output(output)

    def _reshape_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Reshape model output to match v2 prediction format.
        """
        if self.n_quantiles is None:
            return output

        if output.size(-1) != 1:
            raise ValueError(
                "Quantile output expects a single target channel, but got "
                f"{output.size(-1)}."
            )

        batch_size = output.shape[0]
        output = output.squeeze(-1)
        return output.reshape(batch_size, self.prediction_length, self.n_quantiles)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the LightTS model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input batch dictionary from the v2 datamodule.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the prediction tensor under ``"prediction"``.
        """
        input_data, target_indices = self._prepare_input_data(x)
        prediction = self._encoder(input_data, target_indices)

        if (
            "target_scale" in x
            and hasattr(self, "transform_output")
            and self.n_quantiles is None
        ):
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
