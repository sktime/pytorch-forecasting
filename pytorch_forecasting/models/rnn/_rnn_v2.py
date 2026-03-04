"""
Recurrent Network (LSTM/GRU) model for PyTorch Forecasting v2.
---------------------------------------------------------------
"""

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class RecurrentNetwork_v2(TslibBaseModel):
    """
    Recurrent Network model for time series forecasting.

    Supports LSTM and GRU cell types. Encodes the input sequence
    using a recurrent layer and projects the final hidden state
    to the prediction horizon.

    Parameters
    ----------
    loss : nn.Module
        Loss function for training.
    cell_type : str, optional
        Recurrent cell type, either "LSTM" or "GRU". Default is "LSTM".
    hidden_size : int, optional
        Number of features in the hidden state. Default is 64.
    rnn_layers : int, optional
        Number of recurrent layers. Default is 2.
    dropout : float, optional
        Dropout rate between RNN layers. Default is 0.1.
    logging_metrics : list[nn.Module], optional
        Metrics to log during training. Default is None.
    optimizer : str or Optimizer, optional
        Optimizer to use. Default is "adam".
    optimizer_params : dict, optional
        Parameters for the optimizer. Default is None.
    lr_scheduler : str, optional
        Learning rate scheduler. Default is None.
    lr_scheduler_params : dict, optional
        Parameters for the scheduler. Default is None.
    metadata : dict, optional
        Metadata from TslibDataModule. Default is None.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.rnn._rnn_pkg_v2 import RecurrentNetwork_pkg_v2

        return RecurrentNetwork_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        cell_type: str = "LSTM",
        hidden_size: int = 64,
        rnn_layers: int = 2,
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

        assert cell_type in (
            "LSTM",
            "GRU",
        ), f"cell_type must be 'LSTM' or 'GRU', got '{cell_type}'"

        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = dropout

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self.n_quantiles = None
        if isinstance(loss, QuantileLoss):
            self.n_quantiles = len(loss.quantiles)

        self._init_network()

    def _init_network(self):
        """Initialize the RNN network layers."""

        input_size = self.cont_dim + self.target_dim

        if self.cell_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.rnn_layers,
                dropout=self.dropout if self.rnn_layers > 1 else 0,
                batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.rnn_layers,
                dropout=self.dropout if self.rnn_layers > 1 else 0,
                batch_first=True,
            )

        if self.n_quantiles is not None:
            output_size = self.prediction_length * self.n_quantiles
        else:
            output_size = self.prediction_length * self.target_dim

        self.output_projector = nn.Linear(self.hidden_size, output_size)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the RecurrentNetwork model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing input tensors.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing output tensors with key "prediction".
        """
        available_features = []

        if "history_cont" in x and x["history_cont"].size(-1) > 0:
            available_features.append(x["history_cont"])

        if "history_target" in x and x["history_target"].size(-1) > 0:
            available_features.append(x["history_target"])

        if not available_features:
            raise ValueError("No valid input features found in input dictionary.")

        input_data = torch.cat(available_features, dim=-1)

        rnn_out, _ = self.rnn(input_data)

        last_hidden = rnn_out[:, -1, :]

        output = self.output_projector(last_hidden)

        if self.n_quantiles is not None:
            output = output.reshape(-1, self.prediction_length, self.n_quantiles)
        else:
            output = output.reshape(-1, self.prediction_length, self.target_dim)

        if "target_scale" in x and hasattr(self, "transform_output"):
            output = self.transform_output(output, x["target_scale"])

        return {"prediction": output}
