"""
xLSTMTime model for PyTorch Forecasting v2.
"""

from typing import Literal, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.layers import SeriesDecomposition, mLSTMNetwork, sLSTMNetwork
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class xLSTMTime(BaseModel):
    """
    xLSTMTime is a long-term time series forecasting architecture built on the
    extended LSTM (xLSTM) design, incorporating either the scalar-memory
    stabilized LSTM (sLSTM) or the matrix-memory mLSTM variant.

    Based on https://arxiv.org/pdf/2407.10240 and https://github.com/muslehal/xLSTMTime
    """

    @classmethod
    def _pkg(cls):
        """Package for the model."""
        from pytorch_forecasting.models.xlstm._xlstm_pkg_v2 import xLSTMTime_pkg_v2

        return xLSTMTime_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        hidden_size: int = 32,
        xlstm_type: Literal["slstm", "mlstm"] = "slstm",
        num_layers: int = 1,
        decomposition_kernel: int = 25,
        input_projection_size: int | None = None,
        dropout: float = 0.1,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self.metadata = metadata or {}

        if xlstm_type not in ["slstm", "mlstm"]:
            raise ValueError("xlstm_type must be either 'slstm' or 'mlstm'")

        self.max_encoder_length = self.metadata["max_encoder_length"]
        self.max_prediction_length = self.metadata["max_prediction_length"]
        self.input_size = self.metadata["encoder_cont"] + 1

        self.hidden_size = hidden_size
        self.xlstm_type = xlstm_type
        self.input_projection_size = input_projection_size or hidden_size

        self.n_quantiles = 1
        if hasattr(loss, "quantiles") and loss.quantiles is not None:
            self.n_quantiles = len(loss.quantiles)

        output_dim = self.max_prediction_length * self.n_quantiles

        kernel = min(decomposition_kernel, self.max_encoder_length)
        if kernel % 2 == 0:
            kernel = max(1, kernel - 1)
        self.decomposition = SeriesDecomposition(kernel)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.input_linear = nn.Linear(self.input_size * 2, self.input_projection_size)

        if xlstm_type == "mlstm":
            self.lstm = mLSTMNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=hidden_size,
                dropout=dropout,
            )
        else:
            self.lstm = sLSTMNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=hidden_size,
                dropout=dropout,
            )

        self.output_linear = nn.Linear(hidden_size, output_dim)
        self.instance_norm = nn.InstanceNorm1d(output_dim)

    def _prepare_input(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build encoder input from covariates and past target values."""
        encoder_cont = x["encoder_cont"]
        target_past = x["target_past"]
        if target_past.ndim == 2:
            target_past = target_past.unsqueeze(-1)

        if encoder_cont.size(-1) > 0:
            return torch.cat([encoder_cont, target_past], dim=-1)
        return target_past

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the model."""
        encoder_input = self._prepare_input(x)
        batch_size = encoder_input.size(0)

        seasonal, trend = self.decomposition(encoder_input)
        x_proc = torch.cat([trend, seasonal], dim=-1)
        x_proc = self.input_linear(x_proc)

        x_proc = x_proc.transpose(1, 2)
        x_proc = self.batch_norm(x_proc)
        x_proc = x_proc.transpose(1, 2)

        hidden_states = self.lstm.init_hidden(batch_size, device=x_proc.device)
        x_proc = x_proc.transpose(0, 1)
        output, _ = self.lstm(x_proc, *hidden_states)

        if isinstance(output, tuple):
            output = output[0]
        if output.dim() == 2:
            output = output.unsqueeze(0)

        output = self.output_linear(output[-1])
        output = self.instance_norm(output.unsqueeze(1)).squeeze(1)

        if self.n_quantiles > 1:
            prediction = output.view(
                batch_size, self.max_prediction_length, self.n_quantiles
            )
        else:
            prediction = output.unsqueeze(-1)

        return {"prediction": prediction}
