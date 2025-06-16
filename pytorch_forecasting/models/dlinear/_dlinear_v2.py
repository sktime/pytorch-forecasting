"""
LTSF-DLinear model for Pytorch Forecasting.
-------------------------------------------
"""

#################################################
# NOTE: This is an experimental implementation  #
# of LTSF-DLinear model for PTF v2.             #
# It is an unstable API and subject to change.  #
#################################################

from typing import Any, Optional, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from pytorch_forecasting.layers.decomposition import SeriesDecomposition
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class DLinearModel(TslibBaseModel):
    """
    DLinear: Decomposition Linear Model for Long-Term Time Series Forecasting.

    DLinear decomposes time series into trend and seasonal components and applies
    separate linear layers to each component. The final prediction is the sum of
    both components.

    Parameters
    ----------
    loss: nn.Module
        Loss function for training step.
    moving_avg: int , default=25
        Kernel size for moving average decomposition.
    individual: bool, default=False
        Whether to use individual linear layers for each variate (True) or
        shared layers across all variates (False).
    logging_metrics: Optional[list[nn.Module]], default=None
        List of metrics to log during training, validation, and testing.
    optimizer: Optional[Union[Optimizer, str]], default='adam'
        Optimizer to use for training.
    optimizer_params: Optional[dict], default=None
        Parameters for the optimizer.
    lr_scheduler: Optional[str], default=None
        Learning rate scheduler to use.
    lr_scheduler_params: Optional[dict], default=None
        Parameters for the learning rate scheduler.
    metadata: Optional[dict], default=None
        Metadata for the model from TslibDataModule.

    References
    ----------
    [1] https://arxiv.org/pdf/2205.13504
    [2] https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py

    Notes
    -----
    [1] This implementation supports only continuous features. Categorical variables
        will be accomodated in future versions.
    """

    def __init__(
        self,
        loss: nn.Module,
        moving_avg: int = 25,
        individual: bool = False,
        logging_metrics: Optional[list[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
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
            "DLinearModel is an experimental model implemented on TslibBaseModelV2. "
            "It is an unstable version and may be subject to unannounced changes. "
            "Please use with caution."
        )
        self.moving_avg = moving_avg
        self.individual = individual

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

        self.apply(self._weight_init)

    def _weight_init(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.constant(m.weight.data, 1.0 / self.context_length)
            if m.bias is not None:
                nn.init.constant(m.bias.data, 0.0)

    def _init_network(self):
        """
        Initialise the DLinear model network layer components.
        """

        self.enc_in = self.cont_dim + self.target_dim

        self.decomposition = SeriesDecomposition(self.moving_avg)

        self.n_quantiles = None

        if isinstance(self.loss, QuantileLoss):
            self.n_quantiles = len(self.loss.quantiles)

        output_dim = self.prediction_length

        if self.n_quantiles is not None:
            output_dim = self.prediction_length * self.n_quantiles

        if self.individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()

            for i in range(self.enc_in):
                seasonal_layer = nn.Linear(self.context_length, output_dim)
                trend_layer = nn.Linear(self.context_length, output_dim)

                self.linear_seasonal.append(seasonal_layer)
                self.linear_trend.append(trend_layer)
        else:
            self.linear_seasonal = nn.Linear(self.context_length, output_dim)
            self.linear_trend = nn.Linear(self.context_length, output_dim)

    def _encoder(self, x: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        """
        Encoder the input time series through decompoosition and linear layers.

        Parameters
        ----------
        x: torch.Tensor
            Input data fed into the encoder.
        target_indices: torch.Tensor
            Indices of target features to be extracted from the output. If None, all features are returned.

        Returns
        -------
        output: torch.Tensor
            Encoded output tensor of shape (batch_size, prediction_length, n_features)
        """  # noqa: E501

        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output, trend_output = self._process_individual_features(
                seasonal_init, trend_init
            )  # noqa: E501
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        output = seasonal_output + trend_output

        if target_indices is not None:
            output = output[:, target_indices, :]

        output = self._reshape_output(output)

        return output

    def _process_individual_features(
        self, seasonal_init: torch.Tensor, trend_init: torch.Tensor
    ):  # noqa: E501
        """
        Process features individually when self.individual=True.

        Parameters
        ----------
        seasonal_init: Seasonal component tensor
        trend_init: Trend component tensor

        Returns
        -------
            tuple: (seasonal_output, trend_output)
        """
        # Determine output dimension
        if self.n_quantiles is not None:
            output_dim = self.prediction_length * self.n_quantiles
        else:
            output_dim = self.prediction_length

        # Initialize output tensors
        # same batch_size and n_features for both seasonal and trend
        batch_size, n_features, _ = seasonal_init.shape
        seasonal_output = torch.zeros(
            (batch_size, n_features, output_dim),
            dtype=seasonal_init.dtype,
            device=seasonal_init.device,
        )
        trend_output = torch.zeros(
            (batch_size, n_features, output_dim),
            dtype=trend_init.dtype,
            device=trend_init.device,
        )

        # Apply individual linear layers
        for i in range(self.enc_in):
            seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
            trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])

        return seasonal_output, trend_output

    def _reshape_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Reshape output tensor for quantile predictions.

        Parameters
        ----------
        output: torch.Tensor
            Output tensor from the encoder, expected to be of shape
            (batch_size, n_features, prediction_length) or
            (batch_size, n_features, prediction_length, n_quantiles).
        Returns
        -------
        output: torch.Tensor
            Reshaped tensor (batch_size, prediction_length, n_features, n_quantiles)
            or (batch_size, prediction_length, n_features) if n_quantiles is None.
        """
        if self.n_quantiles is not None:
            batch_size, n_features = output.shape[0], output.shape[1]
            output = output.reshape(
                batch_size, n_features, self.prediction_length, self.n_quantiles
            )
            output = output.permute(0, 2, 1, 3)  # (batch, time, features, quantiles)
        else:
            output = output.permute(0, 2, 1)  # (batch, time, features)

        # univariate forecasting
        if self.target_dim == 1 and output.shape[-1] == 1:
            output = output.squeeze(-1)

        return output

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the DLinear model.

        Parameters
        ----------
        x: dict[str, torch.Tensor]
            Dictionary containing input tensors.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing output tensors. These can include
            - predictions: Prediction_output of shape (batch_size, prediction_length, target_dim)
            - attention_weights: Optionally, output attention weights
        """  # noqa: E501
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
