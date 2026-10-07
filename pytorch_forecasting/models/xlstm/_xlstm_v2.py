"""
xLSTMTime model for PyTorch Forecasting v2.
"""

from typing import Literal, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.layers import SeriesDecomposition, mLSTMNetwork, sLSTMNetwork
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class xLSTMTime_v2(BaseModel):
    """
    xLSTMTime is a long-term time series forecasting architecture built on the
    extended LSTM (xLSTM) design, incorporating either the scalar-memory
    stabilized LSTM (sLSTM) or the matrix-memory mLSTM variant.

    Based on https://arxiv.org/pdf/2407.10240 and https://github.com/muslehal/xLSTMTime

    Parameters
    ----------
    loss : nn.Module
        Loss (and evaluation metric) used during training.
    hidden_size : int, default 32
        Hidden size of the xLSTM network; also used by batch norm / LSTM internals.
    xlstm_type : {"slstm", "mlstm"}, default "slstm"
        Specifies which xLSTM variant to use:

        - "slstm": stabilized LSTM with scalar memory,
        - "mlstm": matrix-memory variant for higher capacity and scalability.

    num_layers : int, default 1
        Number of recurrent layers in the sLSTM or mLSTM network.
    decomposition_kernel : int, default 25
        Kernel size for series decomposition into trend and seasonal components.
    input_projection_size : int, optional
        If specified, the encoded input (trend + seasonal) is projected to this size
        before being fed to the xLSTM; otherwise equals hidden_size.
    dropout : float, default 0.1
        Dropout rate applied within the recurrent layers.
    logging_metrics : list of nn.Module, optional
        Metrics logged during training / validation / testing.
    optimizer : Optimizer or str, optional
        Optimizer used for training.
    optimizer_params : dict, optional
        Parameters for the optimizer.
    lr_scheduler : str, optional
        Learning rate scheduler name.
    lr_scheduler_params : dict, optional
        Parameters for the learning rate scheduler.
    metadata : dict, optional
        Metadata from the encoder-decoder datamodule. Used to derive
        ``input_size`` (``encoder_cont + 1`` for ``target_past``) and
        ``output_size`` (``max_prediction_length``, times ``n_quantiles``
        when using ``QuantileLoss``).

    """

    @classmethod
    def _pkg(cls):
        """Package for the model."""
        from pytorch_forecasting.models.xlstm._xlstm_pkg_v2 import xLSTMTime_pkg_v2

        return xLSTMTime_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        hidden_size: int = 32,
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
        **kwargs,
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
        self.xlstm_type = xlstm_type
        self.hidden_size = hidden_size
        self.input_projection_size = input_projection_size or self.hidden_size
        self.decomposition_kernel = decomposition_kernel
        self.num_layers = num_layers
        self.dropout = dropout
        self.metadata = metadata or {}

        if self.xlstm_type not in ["slstm", "mlstm"]:
            raise ValueError(
                "Error in xLSTMTime: xlstm_type must be either 'slstm' or 'mlstm'"
            )

        self.max_encoder_length = self.metadata["max_encoder_length"]
        self.max_prediction_length = self.metadata["max_prediction_length"]
        self.input_size = self.metadata["encoder_cont"] + 1

        self.n_quantiles = 1
        if hasattr(loss, "quantiles") and loss.quantiles is not None:
            self.n_quantiles = len(loss.quantiles)

        self.output_size = self.max_prediction_length * self.n_quantiles

        # self.decomposition = SeriesDecomposition(kernel)
        self.decomposition = SeriesDecomposition(self.decomposition_kernel)
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)

        self.input_linear = nn.Linear(self.input_size * 2, self.input_projection_size)

        if xlstm_type == "mlstm":
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

        self.output_linear = nn.Linear(self.hidden_size, self.output_size)
        self.instance_norm = nn.InstanceNorm1d(self.output_size)

    def _encoder_features(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build encoder input from covariates and past target values."""
        encoder_cont = x["encoder_cont"]
        target_past = x["target_past"]
        if target_past.ndim == 2:
            target_past = target_past.unsqueeze(-1)

        # In v1, the target lived inside ``encoder_cont``. In v2 encoder-decoder
        # batches the target is separate as ``target_past``, so concatenate it.
        if encoder_cont.size(-1) > 0:
            return torch.cat([encoder_cont, target_past], dim=-1)
        return target_past

    def forward(
        self,
        x: dict[str, torch.Tensor],
        hidden_states: tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward Pass for the model."""
        encoder_cont = self._encoder_features(x)
        batch_size, seq_len, n_features = encoder_cont.shape

        seasonal, trend = self.decomposition(encoder_cont)

        x = torch.cat([trend, seasonal], dim=-1)

        x = self.input_linear(x)

        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)

        if hidden_states is None:
            hidden_states = self.lstm.init_hidden(batch_size, device=x.device)

        x = x.transpose(0, 1)
        output, hidden_states = self.lstm(x, *hidden_states)

        if isinstance(output, tuple):
            output = output[0]

        if output.dim() == 2:
            output = output.unsqueeze(0)

        output = self.output_linear(output)

        output = output.transpose(1, 2)
        output = self.instance_norm(output)
        output = output.transpose(1, 2)

        output = output[0, ..., : self.output_size]

        # reshape to (batch, horizon, n_quantiles) when using QuantileLoss
        if self.n_quantiles > 1:
            prediction = output.view(
                batch_size, self.max_prediction_length, self.n_quantiles
            )
        else:
            prediction = output.unsqueeze(-1)

        return {"prediction": prediction}
