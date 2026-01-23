########################################################################################
# Disclaimer: This implementation is based on the new version of data pipeline and is
# experimental, please use with care.
########################################################################################

from typing import Any, Literal , Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_model_v2 import BaseModel
from pytorch_forecasting.models.nn import get_rnn


class DeepAR(BaseModel):
    """
    DeepAR: Probabilistic forecasting with autoregressive recurrent networks.

    Parameters
    ----------
    loss : nn.Module
        Loss function to use.
    logging_metrics : list[nn.Module], optional
        Metrics to log during training.
    optimizer : Union[Optimizer, str], optional
        Optimizer to use. Defaults to "adam".
    optimizer_params : dict, optional
        Parameters for the optimizer.
    lr_scheduler : str, optional
        Learning rate scheduler.
    lr_scheduler_params : dict, optional
        Parameters for the learning rate scheduler.
    cell_type : Literal["LSTM", "GRU"], optional
        Recurrent cell type ["LSTM", "GRU"]. Defaults to "LSTM".
    hidden_size : int, optional
        Hidden recurrent size. Defaults to 10.
    rnn_layers : int, optional
        Number of RNN layers. Defaults to 2.
    dropout : float, optional
        Dropout in RNN layers. Defaults to 0.1.
    metadata : dict, optional
        Metadata from the DataModule.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.deepar.__deepar_pkg_v2 import (
            DeepAR_pkg_v2,
        )

        return DeepAR_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        cell_type: Literal["LSTM", "GRU"] = "LSTM",
        hidden_size: int = 10,
        rnn_layers: int = 2,
        dropout: float = 0.1,
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
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.metadata = metadata or {}

        self.max_encoder_length = self.metadata.get("max_encoder_length", 0)
        self.max_prediction_length = self.metadata.get("max_prediction_length", 0)
        
        self.encoder_cont_dim = self.metadata.get("encoder_cont", 0)
        self.encoder_cat_dim = self.metadata.get("encoder_cat", 0)
        self.decoder_cont_dim = self.metadata.get("decoder_cont", 0)
        self.decoder_cat_dim = self.metadata.get("decoder_cat", 0)
        
        self.target_dim = self.metadata.get("target_dim", 1) 

        rnn_class = get_rnn(cell_type)
        input_size = self.encoder_cont_dim + self.encoder_cat_dim
        
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0,
            batch_first=True,
        )

        self.distribution_projector = nn.Linear(
            hidden_size, len(self.loss.distribution_arguments) * self.target_dim
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the DeepAR model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing input tensors:
            - encoder_cont
            - encoder_cat
            - decoder_cont
            - decoder_cat
            - encoder_lengths
            - decoder_lengths

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing output tensors:
            - prediction: Distribution parameters
        """
        encoder_input = torch.cat([x["encoder_cont"], x["encoder_cat"]], dim=2)
        if self.cell_type == "LSTM":
            _, (h_n, c_n) = self.rnn(encoder_input)
        else:
            _, h_n = self.rnn(encoder_input)
        
        decoder_input = torch.cat([x["decoder_cont"], x["decoder_cat"]], dim=2)
        
        if self.cell_type == "LSTM":
            decoder_output, _ = self.rnn(decoder_input, (h_n, c_n))
        else: 
            decoder_output, _ = self.rnn(decoder_input, h_n)
            
        prediction = self.distribution_projector(decoder_output)
        
        if self.target_dim > 1:
            prediction = prediction.view(
                prediction.size(0), 
                prediction.size(1), 
                self.target_dim, 
                len(self.loss.distribution_arguments)
            )

        return {"prediction": prediction}
