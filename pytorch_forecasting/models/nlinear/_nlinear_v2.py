"""
LTSF-NLinear model for PyTorch Forecasting.
-------------------------------------------
"""

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import DistributionLoss, QuantileLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class NLinear(TslibBaseModel):
    """
    NLinear: Normalization-Linear model for long-term time-series forecasting.

    NLinear normalizes each series by subtracting the last observed value,
    applies a linear projection from context length to prediction length,
    and then restores scale by adding the last observed value back.

    This v2 implementation is intentionally narrow and follows the paper's
    target-history-only formulation. It supports a single target variable with
    a fixed context length and does not currently support exogenous,
    categorical, static, or future-known features.

    References
    ----------
    [1] Ailing Zeng, Muxi Chen, Lei Zhang & Qiang Xu.
        *Are Transformers Effective for Time Series Forecasting?*
        2022. https://arxiv.org/pdf/2205.13504.pdf

    Parameters
    ----------
    loss : nn.Module
        Loss function for training.
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
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ):
        if metadata is None:
            raise ValueError(
                "NLinear requires `metadata` from a fitted datamodule to initialize."
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

        self.n_quantiles: int | None = None

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self._init_network()

    def _init_network(self):
        """Initialize NLinear network layers for single-target input only."""
        if self.context_length <= 0 or self.prediction_length <= 0:
            raise ValueError(
                "NLinear requires positive `context_length` and `prediction_length` "
                "in `metadata`."
            )
        if self.target_dim != 1:
            raise ValueError(
                "NLinear v2 currently supports only a single target variable."
            )
        if isinstance(self.loss, DistributionLoss):
            raise TypeError(
                "NLinear v2 does not support DistributionLoss. "
                "Use QuantileLoss for prediction intervals."
            )

        if isinstance(self.loss, QuantileLoss):
            self.n_quantiles = len(self.loss.quantiles)

        output_dim = self.prediction_length
        if self.n_quantiles is not None:
            output_dim = self.prediction_length * self.n_quantiles

        self.output_dim = output_dim

        self.linear = nn.Linear(self.context_length, output_dim)

    def _encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence and produce forecasts.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, context_length, 1).

        Returns
        -------
        torch.Tensor
            Forecast tensor.
        """
        # Detach to match the original NLinear reference implementation
        # In this implementation history_target is a leaf tensor from the dataloader
        # so this detach has no effect on gradient flow but preserves paper fidelity
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        output = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        output = output + seq_last.expand(-1, output.size(1), -1)

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
            output = output.reshape(
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
        input_data = self._prepare_input_data(x)
        prediction = self._encoder(input_data)

        if "target_scale" in x:
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}

    def _prepare_input_data(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Validate NLinear's narrow input contract and return history_target."""
        unsupported_inputs = {
            "history_cont": "historical continuous features",
            "history_cat": "historical categorical features",
            "future_cont": "future-known continuous features",
            "future_cat": "future-known categorical features",
            "static_categorical_features": "static categorical features",
            "static_continuous_features": "static continuous features",
        }
        for key, description in unsupported_inputs.items():
            value = x.get(key)
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                raise ValueError(
                    "NLinear v2 currently supports target-history-only input and does "
                    f"not accept {description}."
                )

        history_target = x.get("history_target")
        if history_target is None:
            raise ValueError("NLinear requires `history_target` in the input batch.")
        if history_target.ndim != 3:
            raise ValueError(
                "`history_target` must have shape [batch, context_length, 1]."
            )
        if history_target.size(1) != self.context_length:
            raise ValueError(
                "`history_target` length does not match the model context length."
            )
        if history_target.size(-1) != 1:
            raise ValueError(
                "NLinear v2 currently supports only a single target channel."
            )

        return history_target
