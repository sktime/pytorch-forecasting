"""
LTSF-NLinear model for PyTorch Forecasting.
-------------------------------------------
"""

from typing import Any
import warnings

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class NLinear(TslibBaseModel):
    """
    NLinear: Normalization-Linear model for long-term time-series forecasting.

    NLinear normalizes each series by subtracting the last observed value,
    applies a linear projection from context length to prediction length,
    and then restores scale by adding the last observed value back.

    Parameters
    ----------
    loss : nn.Module
        Loss function for training.
    individual : bool, default=False
        Whether to use one linear layer per input channel.
    logging_metrics : list[nn.Module] | None, default=None
        Metrics to log during train/validation/test.
    optimizer : Optimizer | str | None, default="adam"
        Optimizer or optimizer name.
    optimizer_params : dict | None, default=None
        Optimizer keyword arguments.
    lr_scheduler : str | None, default=None
        Learning-rate scheduler name.
    lr_scheduler_params : dict | None, default=None
        Learning-rate scheduler keyword arguments.
    metadata : dict | None, default=None
        Metadata coming from TslibDataModule.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.nlinear._nlinear_pkg_v2 import NLinear_pkg_v2

        return NLinear_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        individual: bool = False,
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

        self.individual = individual
        self.n_quantiles: int | None = None

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self._init_network()

    def _init_network(self):
        """Initialize NLinear network layers."""
        self.enc_in = self.cont_dim + self.target_dim

        if isinstance(self.loss, QuantileLoss):
            self.n_quantiles = len(self.loss.quantiles)
            if self.target_dim != 1:
                raise ValueError(
                    "NLinear currently supports QuantileLoss only for single-target "
                    "forecasting in v2."
                )

        output_dim = self.prediction_length
        if self.n_quantiles is not None:
            output_dim = self.prediction_length * self.n_quantiles

        self.output_dim = output_dim

        if self.individual:
            self.linear = nn.ModuleList(
                [nn.Linear(self.context_length, output_dim) for _ in range(self.enc_in)]
            )
        else:
            self.linear = nn.Linear(self.context_length, output_dim)

    def _encoder(
        self, x: torch.Tensor, target_indices: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Encode input sequence and produce forecasts.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, context_length, n_features).
        target_indices : torch.Tensor | None
            Target channel indices in the model input channels.

        Returns
        -------
        torch.Tensor
            Forecast tensor.
        """
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        if self.individual:
            batch_size, _, n_features = x.shape
            output = torch.zeros(
                (batch_size, self.output_dim, n_features),
                dtype=x.dtype,
                device=x.device,
            )
            for i in range(n_features):
                output[:, :, i] = self.linear[i](x[:, :, i])
        else:
            output = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        output = output + seq_last.expand(-1, output.size(1), -1)

        if target_indices is not None:
            output = output[:, :, target_indices]

        return self._reshape_output(output)

    def _reshape_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Reshape output tensor for quantile predictions.

        Parameters
        ----------
        output : torch.Tensor
            Output tensor from the encoder.

        Returns
        -------
        torch.Tensor
            Reshaped output tensor.
        """
        if self.n_quantiles is not None:
            if output.size(-1) != 1:
                raise ValueError(
                    "Quantile output expects a single target channel, but got "
                    f"{output.size(-1)}."
                )
            batch_size = output.shape[0]
            output = output.squeeze(-1).reshape(
                batch_size, self.prediction_length, self.n_quantiles
            )
        return output

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the NLinear model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing model input tensors.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing output prediction tensor.
        """
        input_data, target_indices = self._prepare_input_data(x)
        prediction = self._encoder(input_data, target_indices)

        if "target_scale" in x and hasattr(self, "transform_output"):
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}

    def _prepare_input_data(self, x: dict[str, torch.Tensor]):
        """Prepare input data and target indices for model input."""
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
        target_indices = (
            torch.tensor(target_indices, dtype=torch.long, device=input_data.device)
            if target_indices
            else None
        )

        return input_data, target_indices
