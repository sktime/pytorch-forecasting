"""
xLSTM Time Series Forecasting Model - v2 Implementation
--------------------------------------------------------
"""

from typing import Any, Literal, Optional, Union
import warnings as warn

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class xLSTM(BaseModel):
    """
    An implementation of xLSTMTime model for v2 of pytorch-forecasting.

    xLSTMTime is a long‑term time series forecasting architecture built on the
    extended LSTM (xLSTM) design, incorporating either the scalar-memory
    stabilized LSTM (sLSTM) or the matrix-memory mLSTM variant. This model
    enhances classical LSTM by adding exponential gating and richer memory
    dynamics, and combines series decomposition and normalization layers to
    produce robust forecasts over extended horizons.

    Parameters
    ----------
    loss: nn.Module
        Loss function to use for training.
    input_size: int, None
        Number of input features for the encoder. If None it instantiates to
        the number of continuous features.
    hidden_size: int, default=128
        Hidden size of the xLSTM network; also used by batch norm / LSTM internals.
    xlstm_type: {"slstm", "mlstm"}, default="slstm"
        Specifies which xLSTM variant to use:
        - "slstm": stabilized LSTM with scalar memory,
        - "mlstm": matrix-memory variant for higher capacity and scalability.
    num_layers: int, default=1
        Number of recurrent layers in the sLSTM or mLSTM network.
    decomposition_kernel: int, default=25
        Kernel size for series decomposition into trend and seasonal components.
    input_projection_size: int, optional
        If specified, the encoded input (trend + seasonal) is projected to this size
        before being fed to the xLSTM; otherwise equals hidden_size.
    dropout: float, default=0.1
        Dropout rate applied within the recurrent layers.
    logging_metrics: Optional[list[nn.Module]], default=None
        List of metrics to log during training, validation, and testing.
    optimizer: Optional[Union[Optimizer, str]], default='adam'
        Optimizer to use for training. Can be a string name or an instance of an optimizer.
    optimizer_params: Optional[dict], default=None
        Parameters for the optimizer. If None, default parameters for the optimizer will be used.
    lr_scheduler: Optional[str], default=None
        Learning rate scheduler to use. If None, no scheduler is used.
    lr_scheduler_params: Optional[dict], default=None
        Parameters for the learning rate scheduler. If None, default parameters for the scheduler will be used.
    metadata: Optional[dict], default=None
        Metadata for the model from TslibDataModule. This can include information about the dataset,
        such as the number of time steps, number of features, etc. It is used to initialize the model
        and ensure it is compatible with the data being used.

    References
    ----------
    [1] https://arxiv.org/pdf/2407.10240
    [2] https://github.com/muslehal/xLSTMTime

    Notes
    -----
    [1] This implementation handles only continuous variables in the context length. Categorical variables
    support will be added in the future.
    [2] The `xLSTMTime` model obtains many of its attributes from the `TslibBaseModel` class, which is a base class
    where a lot of the boiler plate code for metadata handling and model initialization is implemented.
    """  # noqa: E501

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.xlstm._xlstm_pkg_v2 import xLSTM_pkg_v2

        return xLSTM_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        input_size: int = None,
        hidden_size: int = 128,
        xlstm_type: Literal["slstm", "mlstm"] = "slstm",
        num_layers: int = 1,
        decomposition_kernel: int = 25,
        input_projection_size: int | None = None,
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
        self.metadata = metadata
        self.target_dim = self.metadata["target"]
        self.cont_dim = self.metadata["encoder_cont"]
        self.prediction_length = self.metadata["max_prediction_length"]

        self.input_size = input_size or self.cont_dim
        self.hidden_size = hidden_size

        if xlstm_type not in ["slstm", "mlstm"]:
            raise ValueError("xlstm_type must be either 'slstm' or 'mlstm'")
        self.xlstm_type = xlstm_type

        self.num_layers = num_layers
        self.decomposition_kernel = decomposition_kernel
        self.input_projection_size = input_projection_size or hidden_size
        self.dropout = dropout
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self._init_network()

    def _init_network(self):
        """
        Initialize the network for xLSTMTime's architecture.
        """
        from pytorch_forecasting.layers import (
            SeriesDecomposition,
            mLSTMNetwork,
            sLSTMNetwork,
        )

        self.decomposition = SeriesDecomposition(self.decomposition_kernel)
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)

        self.input_linear = nn.Linear(self.input_size * 2, self.input_projection_size)

        if self.xlstm_type == "mlstm":
            self.lstm = mLSTMNetwork(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=self.hidden_size,
                dropout=self.dropout,
            )
        else:  # slstm
            self.lstm = sLSTMNetwork(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=self.hidden_size,
                dropout=self.dropout,
            )

        self.output_size = self.prediction_length * self.target_dim

        self.output_linear = nn.Linear(self.hidden_size, self.output_size)

        self.instance_norm = nn.InstanceNorm1d(self.output_size)

    def _forecast(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the xLSTMTime model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input data.

        Returns
        -------
        torch.Tensor
            Model predictions.
        """
        encoder_cont = x["encoder_cont"]

        batch_size, seq_len, n_features = encoder_cont.shape

        seasonal, trend = self.decomposition(encoder_cont)

        decomposed = torch.cat([trend, seasonal], dim=-1)

        x_proj = self.input_linear(decomposed)

        x_proj = x_proj.transpose(1, 2)
        x_proj = self.batch_norm(x_proj)
        x_proj = x_proj.transpose(1, 2)

        hidden_states = self.lstm.init_hidden(batch_size, device=x_proj.device)

        x_proj = x_proj.transpose(0, 1)

        output, hidden_states = self.lstm(x_proj, *hidden_states)

        if isinstance(output, tuple):
            output = output[0]

        if output.dim() == 2:
            output = output.unsqueeze(0)

        output = self.output_linear(output)

        output = output.transpose(1, 2)
        output = self.instance_norm(output)
        output = output.transpose(1, 2)

        output = output.view(batch_size, self.prediction_length, self.target_dim)
        return output

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the xLSTMTime model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input data.

        Returns
        -------
        dict[str, torch.Tensor]
            Model predictions.
        """
        out = self._forecast(x)
        prediction = out[:, : self.prediction_length, :]
        return {"prediction": prediction}
