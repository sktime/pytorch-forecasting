"""
TSMixer model for PyTorch Forecasting.
-------------------------------------------
"""

#################################################
# NOTE: This is an experimental implementation  #
# of TSMixer for PyTorch Forecasting v2.         #
# It is an unstable API and subject to change.  #
#################################################

from typing import Any
import warnings

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class TSMixerBlock(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        num_features: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Linear(sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sequence_length),
            nn.Dropout(dropout),
        )

        self.channel = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


class TSMixer(TslibBaseModel):
    """
    TSMixer: An All-MLP Architecture for Time Series Forecasting.

    TSMixer is a lightweight time series forecasting model made of stacked
    multilayer perceptrons.

    Parameters
    ----------
    loss : nn.Module
        Loss function for training step.
    d_model : int, default=64
        Hidden dimension of the temporal and channel mixing MLPs.
    e_layers : int, default=2
        Number of stacked TSMixer blocks.
    dropout : float, default=0.1
        Dropout probability used within the mixer blocks.
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
    [1] TSMixer: An All-MLP Architecture for Time Series Forecasting (https://arxiv.org/abs/2303.06053).
    [2] https://github.com/thuml/Time-Series-Library/blob/main/models/TSMixer.py
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.tsmixer._tsmixer_pkg_v2 import TSMixer_pkg_v2

        return TSMixer_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        d_model: int = 64,
        e_layers: int = 2,
        dropout: float = 0.1,
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

        warnings.warn(
            "TSMixer is an experimental model implemented on TslibBaseModelV2. "
            "It is an unstable version and may be subject to unannounced changes. "
            "Please use with caution."
        )

        self.d_model = d_model
        self.e_layers = e_layers
        self.dropout = dropout

        self.save_hyperparameters(
            ignore=["loss", "logging_metrics", "metadata"]
        )

        self._init_network()

    def _init_network(self):
        """Initialize the TSMixer network components."""

        self.enc_in = self.cont_dim + self.target_dim

        self.model = nn.ModuleList(
            [
                TSMixerBlock(
                    sequence_length=self.context_length,
                    num_features=self.enc_in,
                    hidden_dim=self.d_model,
                    dropout=self.dropout,
                )
                for _ in range(self.e_layers)
            ]
        )

        self.projection = nn.Linear(
            self.context_length,
            self.prediction_length,
        )
    
    def _encoder(self, x: torch.Tensor, target_indices: torch.Tensor | None) -> torch.Tensor:
        """
        Encode the input time series through the TSMixer blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input data to the encoder via tensor of shape
            (batch_size, context_length, n_features).
        target_indices : torch.Tensor or None
            Indices of target features to extract from the output.
            If None, all features are returned.

        Returns
        -------
        torch.Tensor
            Forecast tensor of shape
            (batch_size, prediction_length, n_targets).
        """

        for block in self.model:
            x = block(x)

        output = self.projection(x.transpose(1, 2))

        if target_indices is not None:
            output = output[:, target_indices, :]

        output = output.transpose(1, 2)

        return output

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
