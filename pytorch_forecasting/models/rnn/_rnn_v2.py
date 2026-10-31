"""
Recurrent Network (LSTM/GRU) model for PyTorch Forecasting v2.
---------------------------------------------------------------
"""

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class RNN(BaseModel):
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
        Metadata from EncoderDecoderTimeSeriesDataModule. Default is None.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.rnn._rnn_pkg_v2 import RNN_pkg_v2

        return RNN_pkg_v2

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
        )

        assert cell_type in (
            "LSTM",
            "GRU",
        ), f"cell_type must be 'LSTM' or 'GRU', got '{cell_type}'"

        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.metadata = metadata

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self.max_encoder_length = metadata["max_encoder_length"]
        self.max_prediction_length = metadata["max_prediction_length"]
        self.encoder_cont = metadata["encoder_cont"]
        self.encoder_cat = metadata["encoder_cat"]
        self.input_dim = self.encoder_cont + self.encoder_cat

        self.n_quantiles = None
        if isinstance(loss, QuantileLoss):
            self.n_quantiles = len(loss.quantiles)

        self._init_network()

    def _init_network(self):
        """Initialize the RNN network layers."""
        if self.cell_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=max(1, self.input_dim),
                hidden_size=self.hidden_size,
                num_layers=self.rnn_layers,
                dropout=self.dropout if self.rnn_layers > 1 else 0,
                batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                input_size=max(1, self.input_dim),
                hidden_size=self.hidden_size,
                num_layers=self.rnn_layers,
                dropout=self.dropout if self.rnn_layers > 1 else 0,
                batch_first=True,
            )

        if self.n_quantiles is not None:
            output_size = self.max_prediction_length * self.n_quantiles
        else:
            output_size = self.max_prediction_length

        self.output_projector = nn.Linear(self.hidden_size, output_size)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the RNN model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing input tensors.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing output tensors with key "prediction".
        """
        batch_size = x["encoder_cont"].shape[0]

        encoder_cont = x.get(
            "encoder_cont",
            torch.zeros(batch_size, self.max_encoder_length, 0, device=self.device),
        )
        encoder_cat = x.get(
            "encoder_cat",
            torch.zeros(batch_size, self.max_encoder_length, 0, device=self.device),
        )

        input_data = torch.cat([encoder_cont, encoder_cat], dim=-1)

        if input_data.size(-1) == 0:
            input_data = torch.zeros(
                batch_size, self.max_encoder_length, 1, device=self.device
            )

        rnn_out, _ = self.rnn(input_data)
        last_hidden = rnn_out[:, -1, :]
        output = self.output_projector(last_hidden)

        if self.n_quantiles is not None:
            output = output.reshape(-1, self.max_prediction_length, self.n_quantiles)
        else:
            output = output.reshape(-1, self.max_prediction_length, 1)

        return {"prediction": output}
