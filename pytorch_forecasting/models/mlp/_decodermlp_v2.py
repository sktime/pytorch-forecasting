"""
DecoderMLP v2 model for time series forecasting.

MLP that predicts output only based on information available in the decoder
(known future covariates).
"""

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import MAE, Metric
from pytorch_forecasting.models.base._base_model_v2 import BaseModel
from pytorch_forecasting.models.mlp.submodules import FullyConnectedModule


class DecoderMLP(BaseModel):
    """MLP on the decoder - v2 implementation.

    MLP that predicts output only based on information available in the decoder,
    i.e., known future covariates. This is a simple yet effective baseline model
    for time series forecasting when strong future covariates are available.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.mlp._decodermlp_pkg_v2 import (
            DecoderMLP_pkg_v2,
        )

        return DecoderMLP_pkg_v2

    def __init__(
        self,
        loss: Metric | None = None,
        *,
        activation_class: str = "ReLU",
        hidden_size: int = 300,
        n_hidden_layers: int = 3,
        dropout: float = 0.1,
        norm: bool = True,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ):
        if loss is None:
            loss = MAE()

        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self.metadata = metadata

        # Extract dimensions from metadata
        self.prediction_length = metadata["max_prediction_length"]
        self.decoder_cont_dim = metadata.get("decoder_cont", 0)
        self.decoder_cat_dim = metadata.get("decoder_cat", 0)
        self.target_dim = metadata.get("target", 1)

        # Determine quantile count from loss
        self.n_quantiles = 1
        if hasattr(loss, "quantiles") and loss.quantiles is not None:
            self.n_quantiles = len(loss.quantiles)

        # Compute input/output sizes
        input_size = self.decoder_cont_dim + self.decoder_cat_dim
        if input_size == 0:
            # Fallback: use encoder_cont as input dimension if no decoder
            # features are available (the model still needs some input)
            input_size = metadata.get("encoder_cont", 1)

        output_size = self.target_dim * self.n_quantiles

        # Build MLP network
        self.mlp = FullyConnectedModule(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            n_hidden_layers=n_hidden_layers,
            activation_class=getattr(nn, activation_class),
            dropout=dropout,
            norm=norm,
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError
