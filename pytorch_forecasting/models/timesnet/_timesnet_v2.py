"""
TimesNet model for PyTorch Forecasting.
---------------------------------------
"""

#################################################
# NOTE: This is an experimental implementation  #
# of TimesNet for PyTorch Forecasting v2.       #
# It is an unstable API and subject to change.  #
#################################################

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers._blocks import TimesBlock
from pytorch_forecasting.layers._embeddings._positional_embedding import (
    PositionalEmbedding,
)
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class TimesNet(TslibBaseModel):
    """
    TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.

    TimesNet transforms 1D time series into 2D tensors based on dominant
    periodicities discovered via Fast Fourier Transform (FFT), then applies
    multi-scale 2D Inception convolutions to jointly model intraperiod- and
    interperiod-variations.

    Parameters
    ----------
    loss : nn.Module
        Loss function for training step.
    d_model : int, default=64
        Hidden embedding dimension of the model.
    d_ff : int, default=64
        Intermediate channel dimension inside the 2D Inception blocks.
    e_layers : int, default=2
        Number of stacked TimesBlock layers.
    top_k : int, default=5
        Number of top dominant periods selected via FFT in each TimesBlock.
    num_kernels : int, default=6
        Number of parallel 2D convolution kernels in each Inception block.
    dropout : float, default=0.1
        Dropout probability applied in the positional embedding layer.
    logging_metrics : Optional[list[nn.Module]], default=None
        List of metrics to log during training, validation and testing.
    optimizer : Optional[Union[Optimizer, str]], default="adam"
        Optimizer used for training.
    optimizer_params : Optional[dict], default=None
        Parameters passed to the optimizer.
    lr_scheduler : Optional[str], default=None
        Learning rate scheduler utilised.
    lr_scheduler_params : Optional[dict], default=None
        Parameters passed to the learning rate scheduler.
    metadata : Optional[dict], default=None
        Metadata for the model from TslibDataModule.

    References
    ----------
    [1] TimesNet: Temporal 2D-Variation Modeling for General Time Series
        Analysis (https://arxiv.org/abs/2210.02186).
    [2] https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.timesnet._timesnet_forecaster_v2 import (
            TimesNetForecaster,
        )

        return TimesNetForecaster

    def __init__(
        self,
        loss: nn.Module,
        d_model: int = 64,
        e_layers: int = 2,
        dropout: float = 0.1,
        d_ff=64,
        top_k=5,
        num_kernels=6,
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

        self.d_model = d_model
        self.e_layers = e_layers
        self.dropout = dropout
        self.d_ff = d_ff
        self.top_k = top_k
        self.num_kernels = num_kernels

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """Initialize the TimesNet network components."""
        self.enc_in = self.cont_dim + self.target_dim
        self.value_embedding = nn.Conv1d(
            self.enc_in,
            self.d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        self.position_embedding = PositionalEmbedding(
            d_model=self.d_model,
            dropout=self.dropout,
            add_x=True,
        )
        self.predict_linear = nn.Linear(
            self.context_length, self.context_length + self.prediction_length
        )

        self.layer_norm = nn.LayerNorm(self.d_model)

        self.model = nn.ModuleList(
            [
                TimesBlock(
                    d_ff=self.d_ff,
                    top_k=self.top_k,
                    num_kernels=self.num_kernels,
                    d_model=self.d_model,
                )
                for _ in range(self.e_layers)
            ]
        )

        self.n_quantiles = None
        output_dim = self.target_dim

        if isinstance(self._loss, QuantileLoss):
            if self.target_dim != 1:
                raise ValueError(
                    "quantile forecasting currently only supports a single target"
                )
            self.n_quantiles = len(self._loss.quantiles)
            output_dim = self.n_quantiles

        self.projection = nn.Linear(self.d_model, output_dim)

    def _encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input time series through embedding, temporal projection,
        stacked TimesBlocks, and output projection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, context_length, enc_in)``.

        Returns
        -------
        torch.Tensor
            Forecast tensor.
                For point forecasts, returns shape
                ``(batch_size, prediction_length, target_dim)``.
                For quantile forecasts, returns shape
                ``(batch_size, prediction_length, n_quantiles)``.
        """
        x = self.value_embedding(x.transpose(1, 2)).transpose(1, 2)
        x = self.position_embedding(x)

        x = self.predict_linear(x.transpose(1, 2)).transpose(1, 2)

        for block in self.model:
            x = self.layer_norm(block(x))

        output = self.projection(x)
        return output[:, -self.prediction_length :, :]

    def _prepare_input_data(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare and concatenate continuous and target history features."""
        available_features = []

        if "history_cont" in x and x["history_cont"].size(-1) > 0:
            available_features.append(x["history_cont"])

        if "history_target" not in x or x["history_target"].size(-1) == 0:
            raise ValueError("No target history found in the input dictionary.")

        available_features.append(x["history_target"])
        return torch.cat(available_features, dim=-1)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the TimesNet model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing input tensors (``history_target``,
            optionally ``history_cont`` and ``target_scale``).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the model prediction under ``"prediction"``.
        """
        input_data = self._prepare_input_data(x)
        prediction = self._encoder(input_data)

        if "target_scale" in x and hasattr(self, "transform_output"):
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
