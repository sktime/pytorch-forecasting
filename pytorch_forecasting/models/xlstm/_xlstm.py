from copy import copy
from typing import Literal, Optional, Union

import torch
from torch import nn

from pytorch_forecasting.layers import SeriesDecomposition, mLSTMNetwork, sLSTMNetwork
from pytorch_forecasting.metrics import SMAPE, Metric
from pytorch_forecasting.models.base_model import AutoRegressiveBaseModel


class xLSTMTime(AutoRegressiveBaseModel):
    """
    xLSTMTime is a long‑term time series forecasting architecture built on the
    extended LSTM (xLSTM) design, incorporating either the scalar-memory
    stabilized LSTM (sLSTM) or the matrix-memory mLSTM variant. This model
    enhances classical LSTM by adding exponential gating and richer memory
    dynamics, and combines series decomposition and normalization layers to
    produce robust forecasts over extended horizons.

    It is based on this paper: https://arxiv.org/pdf/2407.10240 and
    https://github.com/muslehal/xLSTMTime
    """

    @classmethod
    def _pkg(cls):
        """Package for the model."""
        from pytorch_forecasting.models.xlstm._xlstm_pkg import xLSTMTime_pkg

        return xLSTMTime_pkg

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        xlstm_type: Literal["slstm", "mlstm"] = "slstm",
        num_layers: int = 1,
        decomposition_kernel: int = 25,
        input_projection_size: int | None = None,
        dropout: float = 0.1,
        loss: Metric = SMAPE(),
        **kwargs,
    ):
        """
        Initialise the model.

        Parameters
        ----------
        input_size : int
            Number of input continuous features per time step.
        hidden_size : int
            Hidden size of the xLSTM network; also used by batch norm / LSTM internals.
        output_size : int
            Number of output features per time step (forecast horizon).
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
        loss : pytorch_forecasting.metrics.Metric, default SMAPE()
            Loss (and evaluation metric) used during training.
        """
        if "target" in kwargs:
            del kwargs["target"]
        if "target_lags" in kwargs:
            del kwargs["target_lags"]
        self.save_hyperparameters()
        super().__init__(loss=loss, **kwargs)

        if xlstm_type not in ["slstm", "mlstm"]:
            raise ValueError("xlstm_type must be either 'slstm' or 'mlstm'")

        self.xlstm_type = xlstm_type

        self.decomposition = SeriesDecomposition(decomposition_kernel)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        self.input_projection_size = input_projection_size or hidden_size

        self.input_linear = nn.Linear(input_size * 2, self.input_projection_size)

        if xlstm_type == "mlstm":
            self.lstm = mLSTMNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=hidden_size,
                dropout=dropout,
            )
        else:  # slstm
            self.lstm = sLSTMNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=hidden_size,
                dropout=dropout,
            )

        self.output_linear = nn.Linear(hidden_size, output_size)
        self.instance_norm = nn.InstanceNorm1d(output_size)

    def forward(
        self,
        x: dict[str, torch.Tensor],
        hidden_states: tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward Pass for the model."""
        encoder_cont = x["encoder_cont"]
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

        output = output[0, ..., : self.hparams.output_size]
        output = output.unsqueeze(-1)
        return self.to_network_output(prediction=output)

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        """
        Create model from dataset and set parameters related to covariates.

        Parameters
        ----------
        dataset: timeseries dataset
        **kwargs: additional arguments such as hyperparameters for model

        Returns
        -------
            xLSTMTime
        """
        from pytorch_forecasting.data.encoders import NaNLabelEncoder

        assert not isinstance(dataset.target_normalizer, NaNLabelEncoder), (
            "only regression tasks are supported - target must not be categorical"
        )

        new_kwargs = copy(kwargs)
        new_kwargs.update(
            cls.deduce_default_output_parameters(dataset, kwargs, SMAPE())
        )

        return super().from_dataset(dataset, **kwargs)
