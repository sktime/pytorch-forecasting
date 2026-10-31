"""RecurrentNetwork (RNN / LSTM / GRU) model for PyTorch Forecasting v2."""

from typing import Any, Literal

import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import Metric
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class RecurrentNetwork_v2(BaseModel):
    """RecurrentNetwork is a sequential time-series forecasting architecture

    built on standard PyTorch recurrent neural network layers (LSTM or GRU).

    Parameters
    ----------
    loss : Metric
        Loss function used during training.
    cell_type : {"LSTM", "GRU"}, default="LSTM"
        Type of recurrent cell to use.
    hidden_size : int, default=10
        Hidden size of the recurrent layers.
    rnn_layers : int, default=2
        Number of recurrent layers.
    dropout : float, default=0.1
        Dropout rate applied between recurrent layers.
    logging_metrics : list of nn.Module, optional
        Metrics logged during training, validation, and testing.
    optimizer : Optimizer or str, default="adam"
        Optimizer used for training.
    optimizer_params : dict, optional
        Additional parameters for the optimizer.
    lr_scheduler : str, optional
        Learning rate scheduler name.
    lr_scheduler_params : dict, optional
        Parameters for the learning rate scheduler.
    metadata : dict, optional
        Metadata from the data module. Used to derive ``input_size``,
        ``max_encoder_length``, and ``max_prediction_length``.
    """

    @classmethod
    def _pkg(cls):
        """Package container for the model."""
        from pytorch_forecasting.models.rnn._rnn_pkg_v2 import (
            RecurrentNetwork_pkg_v2,
        )

        return RecurrentNetwork_pkg_v2

    def __init__(
        self,
        loss: Metric,
        cell_type: Literal["LSTM", "GRU"] = "LSTM",
        hidden_size: int = 10,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict[str, Any] | None = None,
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
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self.cell_type = cell_type.upper()
        if self.cell_type not in ["LSTM", "GRU"]:
            raise ValueError(
                f"Invalid cell_type: {cell_type}. Supported types are 'LSTM' and 'GRU'."
            )

        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.metadata = metadata or {}

        self.max_encoder_length = self.metadata.get("max_encoder_length", 10)
        self.max_prediction_length = self.metadata.get("max_prediction_length", 1)

        # Calculate input feature dimensions from metadata
        encoder_cont_dim = self.metadata.get("encoder_cont", 0)
        target_dim = 1
        self.input_size = encoder_cont_dim + target_dim

        self.n_quantiles = 1
        quantiles = getattr(loss, "quantiles", None)
        if quantiles is not None and hasattr(quantiles, "__len__"):
            self.n_quantiles = len(quantiles)

        self.output_size = self.max_prediction_length * self.n_quantiles

        rnn_class = nn.LSTM if self.cell_type == "LSTM" else nn.GRU
        self.rnn = rnn_class(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.rnn_layers,
            dropout=self.dropout if self.rnn_layers > 1 else 0.0,
            batch_first=True,
        )

        self.output_projector = nn.Linear(self.hidden_size, self.output_size)

    def _build_input_tensor(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build combined input tensor from continuous features and targets."""
        target_past = x.get("target_past")
        if target_past is not None and target_past.ndim == 2:
            target_past = target_past.unsqueeze(-1)

        encoder_cont = x.get("encoder_cont")
        if encoder_cont is not None and encoder_cont.size(-1) > 0:
            if target_past is not None:
                return torch.cat([encoder_cont, target_past], dim=-1)
            return encoder_cont
        elif target_past is not None:
            return target_past
        else:
            raise KeyError(
                "Neither 'target_past' nor 'encoder_cont' found in input dict."
            )

    def forward(
        self,
        x: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass for RecurrentNetwork_v2.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing input tensors from the DataModule.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing predicted output tensor under key ``prediction``.
        """
        input_tensor = self._build_input_tensor(x)
        batch_size = input_tensor.size(0)

        rnn_out, _ = self.rnn(input_tensor)
        last_hidden = rnn_out[:, -1, :]

        projected = self.output_projector(last_hidden)

        if self.n_quantiles > 1:
            prediction = projected.view(
                batch_size, self.max_prediction_length, self.n_quantiles
            )
        else:
            prediction = projected.unsqueeze(-1)

        return {"prediction": prediction}
