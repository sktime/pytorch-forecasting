"""
LTSF-DLinear model for Pytorch Forecasting.
-------------------------------------------
"""

################################################
# NOTE: This is an experimental implementation #
# of LTSF-DLinear model. It is an unstable API #
################################################

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
    enc_in: int, optional
        Number of input features for the encoder. If not provided,
        it is initialised to the number of continuous features in the dataset.
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
        enc_in: Optional[int] = None,
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

        self.enc_in = enc_in
        self.moving_avg = moving_avg
        self.individual = individual

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """
        Initialise the DLinear model network layer components.
        """

        self.enc_in = self.enc_in or self.cont_dim

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
                self.linear_seasonal.append(nn.Linear(self.context_length, output_dim))
                self.linear_trend.append(nn.Linear(self.context_length, output_dim))

                # initialise the weights for the linear layers
                # begins with a uniform and linear distribution.
                self.linear_seasonal[i].weight = nn.Parameter(
                    (1 / self.context_length)
                    * torch.ones([output_dim, self.context_length])  # noqa: E501
                )

                self.linear_trend[i].weight = nn.Parameter(
                    (1 / self.context_length)
                    * torch.ones([output_dim, self.context_length])  # noqa: E501
                )

        else:
            self.linear_seasonal = nn.Linear(self.context_length, output_dim)
            self.linear_trend = nn.Linear(self.context_length, output_dim)

            self.linear_seasonal.weight = nn.Parameter(
                (1 / self.context_length)
                * torch.ones([output_dim, self.context_length])  # noqa: E501
            )

            self.linear_trend.weight = nn.Parameter(
                (1 / self.context_length)
                * torch.ones([output_dim, self.context_length])  # noqa: E501
            )

    def _encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder the input time series through decompoosition and linear layers.

        Args:
            x (dict[str, torch.Tensor]): Input data

        Returns:
            torch.Tensor: Encoded output tensor of shape (batch_size, prediction_length, n_features)
        """  # noqa: E501

        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = (
            seasonal_init.permute(0, 2, 1),
            trend_init.permute(0, 2, 1),
        )  # noqa: E501

        if self.individual:
            if self.n_quantiles is not None:
                seasonal_output = torch.zeros(
                    [
                        seasonal_init.size(0),
                        seasonal_init.size(1),
                        self.prediction_length * self.n_quantiles,
                    ],  # noqa: E501
                    dtype=seasonal_init.dtype,
                    device=seasonal_init.device,
                )
                trend_output = torch.zeros(
                    [
                        trend_init.size(0),
                        trend_init.size(1),
                        self.prediction_length * self.n_quantiles,
                    ],  # noqa: E501
                    dtype=trend_init.dtype,
                    device=trend_init.device,
                )
            else:
                seasonal_output = torch.zeros(
                    [
                        seasonal_init.size(0),
                        seasonal_init.size(1),
                        self.prediction_length,
                    ],  # noqa: E501
                    dtype=seasonal_init.dtype,
                    device=seasonal_init.device,
                )
                trend_output = torch.zeros(
                    [trend_init.size(0), trend_init.size(1), self.prediction_length],  # noqa: E501
                    dtype=trend_init.dtype,
                    device=trend_init.device,
                )

            for i in range(self.enc_in):
                seasonal_output[:, i, :] = self.linear_seasonal[i](
                    seasonal_init[:, i, :]
                )

                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        output = seasonal_output + trend_output

        if self.n_quantiles is not None:
            batch_size, n_features = output.shape[0], output.shape[1]

            output = output.view(
                batch_size, n_features, self.prediction_length, self.n_quantiles
            )  # noqa: E501
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 1)

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
        feature_mode = self.features

        if feature_mode == "S":
            if "history_target" in x and x["history_target"].size(-1) > 0:
                input_data = x["history_target"]
            else:
                raise ValueError(
                    "For 'S' feature mode, 'history_target' must be provided in the input."  # noqa: E501
                )
        elif feature_mode == "M":
            available_features = []
            if "history_cont" in x and x["history_cont"].size(-1) > 0:
                available_features.append(x["history_cont"])
            if "history_target" in x and x["history_target"].size(-1) > 0:
                available_features.append(x["history_target"])

            if not available_features:
                raise ValueError(
                    "For 'M' feature mode, either 'history_cont' or 'history_target' must be provided in the input."  # noqa: E501
                )
            input_data = torch.cat(available_features, dim=-1)
        else:
            if "history_cont" in x and x["history_cont"].size(-1) > 0:
                input_data = x["history_cont"]
            elif "history_target" in x and x["history_target"].size(-1) > 0:
                input_data = x["history_target"]
            else:
                raise ValueError(
                    "For 'MS' feature mode, either 'history_cont' or 'history_target' must be provided in the input."  # noqa: E501
                )

        prediction = self._encoder(input_data)

        if "target_scale" in x and hasattr(self, "transform_output"):
            prediction = self.transform_output(prediction, x["target_scale"])

        # if hasattr(self, "target_indices") and len(self.target_indices) > 0:
        #     prediction = prediction[..., self.target_indices]

        return {"prediction": prediction}
