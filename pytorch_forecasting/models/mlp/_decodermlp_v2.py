"""
DecoderMLP v2 - MLP forecasting model for pytorch-forecasting v2 pipeline.
"""

from typing import Any
import warnings

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, QuantileLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel
from pytorch_forecasting.models.mlp.submodules import FullyConnectedModule


class DecoderMLP(TslibBaseModel):
    """DecoderMLP v2: Fully-connected MLP for the PTF v2 pipeline."""

    @classmethod
    def _pkg(cls):
        from pytorch_forecasting.models.mlp._decodermlp_pkg import DecoderMLP_pkg

        return DecoderMLP_pkg

    def __init__(
        self,
        loss: nn.Module = None,
        hidden_size: int = 300,
        n_hidden_layers: int = 3,
        dropout: float = 0.1,
        norm: bool = True,
        activation_class: str = "ReLU",
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ):
        if loss is None:
            loss = QuantileLoss()
        if logging_metrics is None:
            logging_metrics = nn.ModuleList(
                [SMAPE(), MAE(), RMSE(), MAPE(), MASE()]
            )

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
            "DecoderMLP is an experimental model implemented on TslibBaseModel v2. "
            "It is an unstable version and may be subject to unannounced changes. "
            "Please use with caution.",
            UserWarning,
        )

        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.dropout = dropout
        self.norm = norm
        self.activation_class = activation_class

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self._init_network()

    def _init_network(self):
        # Input: flatten (prediction_length, target_dim) to 1-D vector
        input_size = self.prediction_length * self.target_dim

        self.n_quantiles = None
        if isinstance(self.loss, QuantileLoss):
            self.n_quantiles = len(self.loss.quantiles)
            output_size = self.prediction_length * self.n_quantiles
        else:
            output_size = self.prediction_length * self.target_dim

        self.mlp = FullyConnectedModule(
            input_size=input_size,
            output_size=output_size,
            hidden_size=self.hidden_size,
            n_hidden_layers=self.n_hidden_layers,
            dropout=self.dropout,
            norm=self.norm,
            activation_class=getattr(nn, self.activation_class),
        )

    def forward(
        self, x: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Forward pass of DecoderMLP v2."""
        if "future_cont" in x and x["future_cont"].size(-1) > 0:
            network_input = x["future_cont"]
        elif "history_target" in x and x["history_target"].size(-1) > 0:
            network_input = x["history_target"][:, -self.prediction_length :, :]
        elif "future_target" in x and x["future_target"].size(-1) > 0:
            network_input = x["future_target"]
        else:
            raise ValueError(
                "DecoderMLP v2 requires 'future_cont', 'history_target', "
                "or 'future_target' in the input batch dictionary."
            )

        batch_size = network_input.size(0)
        flat_input = network_input.reshape(batch_size, -1)
        flat_output = self.mlp(flat_input)

        if self.n_quantiles is not None:
            prediction = flat_output.reshape(
                batch_size, self.prediction_length, self.n_quantiles
            )
        else:
            prediction = flat_output.reshape(
                batch_size, self.prediction_length, self.target_dim
            )

        if "target_scale" in x:
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
