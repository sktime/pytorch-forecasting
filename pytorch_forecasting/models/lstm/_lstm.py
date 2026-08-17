"""Simple autoregressive LSTM model for time series forecasting."""

import torch
import torch.nn as nn

from pytorch_forecasting.data.encoders import MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, MultiHorizonMetric, MultiLoss
from pytorch_forecasting.models.base import AutoRegressiveBaseModel
from pytorch_forecasting.models.nn import LSTM
from pytorch_forecasting.utils import to_list


class LSTMModel(AutoRegressiveBaseModel):
    """
    Autoregressive LSTM model supporting univariate and multivariate forecasting.

    Encodes the history with an LSTM and decodes autoregressively, feeding
    predictions back as inputs at each step.

    Parameters
    ----------
    target : str or list of str
        Target variable name(s).
    target_lags : dict
        Lagged target variable names, passed automatically by ``from_dataset``.
    n_layers : int
        Number of LSTM layers.
    hidden_size : int
        LSTM hidden size.
    dropout : float, optional
        Dropout between LSTM layers (ignored when n_layers=1). Default 0.1.
    loss : MultiHorizonMetric, optional
        Loss function. Defaults to MAE(). Use MultiLoss for multiple targets.
    **kwargs
        Additional arguments passed to
        :py:class:`~pytorch_forecasting.models.base.AutoRegressiveBaseModel`.
    """

    @classmethod
    def _pkg(cls):
        from pytorch_forecasting.models.lstm._lstm_pkg import LSTMModel_pkg

        return LSTMModel_pkg

    def __init__(
        self,
        target: str | list,
        target_lags: dict,
        n_layers: int,
        hidden_size: int,
        dropout: float = 0.1,
        loss: MultiHorizonMetric = None,
        **kwargs,
    ):
        if loss is None:
            loss = MAE()
        kwargs.pop("output_size", None)
        self.save_hyperparameters()
        super().__init__(loss=loss, **kwargs)

        n_targets = len(to_list(target))
        self.lstm = LSTM(
            input_size=n_targets,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout if self.hparams.n_layers > 1 else 0,
            batch_first=True,
        )

        if isinstance(target, str):
            self.output_projector = nn.Linear(self.hparams.hidden_size, 1)
        else:
            self.output_projector = nn.ModuleList(
                [nn.Linear(self.hparams.hidden_size, 1) for _ in target]
            )

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        """
        Create model from a ``TimeSeriesDataSet``.

        Infers output size and wraps loss in ``MultiLoss`` for multiple targets.
        """
        assert not isinstance(dataset.target_normalizer, NaNLabelEncoder) and (
            not isinstance(dataset.target_normalizer, MultiNormalizer)
            or all(
                not isinstance(n, NaNLabelEncoder) for n in dataset.target_normalizer
            )
        ), "LSTMModel only supports continuous targets."

        new_kwargs = cls.deduce_default_output_parameters(
            dataset=dataset, kwargs=kwargs, default_loss=MAE()
        )
        new_kwargs.update(kwargs)
        return super().from_dataset(dataset, **new_kwargs)

    @property
    def target_positions(self) -> torch.LongTensor:
        if hasattr(self, "hparams") and isinstance(self.hparams.target, (list, tuple)):
            return torch.arange(
                len(self.hparams.target), device=self.device, dtype=torch.long
            )
        return torch.tensor([0], device=self.device, dtype=torch.long)

    def encode(self, x: dict[str, torch.Tensor]):
        assert x["encoder_lengths"].min() >= 1
        input_vector = x["encoder_cont"][..., self.target_positions].clone()
        input_vector = torch.roll(input_vector, shifts=1, dims=1)[:, 1:]
        _, hidden_state = self.lstm(
            input_vector,
            lengths=x["encoder_lengths"] - 1,
            enforce_sorted=False,
        )
        return hidden_state

    def decode(self, x: dict[str, torch.Tensor], hidden_state):
        input_vector = x["decoder_cont"][..., self.target_positions].clone()
        input_vector = torch.roll(input_vector, shifts=1, dims=1)
        last_encoder_target = x["encoder_cont"][
            torch.arange(x["encoder_cont"].size(0), device=self.device),
            x["encoder_lengths"] - 1,
        ][..., self.target_positions]
        input_vector[:, 0, :] = last_encoder_target

        if self.training:
            lstm_out, _ = self.lstm(
                input_vector,
                hidden_state,
                lengths=x["decoder_lengths"],
                enforce_sorted=False,
            )
            return self.transform_output(
                self._project(lstm_out), target_scale=x["target_scale"]
            )

        target_pos = self.target_positions

        def decode_one(idx, lagged_targets, hidden_state):
            step = input_vector[:, [idx]].clone()
            step[:, 0, :] = lagged_targets[-1]
            lstm_out, hidden_state = self.lstm(step, hidden_state)
            return self._project(lstm_out[:, 0]), hidden_state

        return self.decode_autoregressive(
            decode_one,
            first_target=input_vector[:, 0, target_pos],
            first_hidden_state=hidden_state,
            target_scale=x["target_scale"],
            n_decoder_steps=input_vector.size(1),
        )

    def _project(self, lstm_out: torch.Tensor):
        if isinstance(self.output_projector, nn.ModuleList):
            return [proj(lstm_out) for proj in self.output_projector]
        return self.output_projector(lstm_out)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden_state = self.encode(x)
        output = self.decode(x, hidden_state)
        return self.to_network_output(prediction=output)
