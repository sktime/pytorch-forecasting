__all__ = ["LSTMModel"]

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from pytorch_forecasting.metrics import MAE, Metric, MultiLoss
from pytorch_forecasting.models.nn import LSTM

from ._base_autoregressive import AutoRegressiveBaseModel


class LSTMModel(AutoRegressiveBaseModel):
    """Simple LSTM model.

    Args:
        target (Union[str, Sequence[str]]):
            Name (or list of names) of target variable(s).

        target_lags (Dict[str, Dict[str, int]]): _description_

        n_layers (int):
            Number of LSTM layers.

        hidden_size (int):
            Hidden size for LSTM model.

        dropout (float, optional):
            Droput probability (<1). Defaults to 0.1.

        input_size (int, optional):
            Input size. Defaults to: inferred from `target`.

        loss (Metric):
            Loss criterion. Can be different for each target in multi-target setting thanks to
            `MultiLoss`. Defaults to `MAE`.

        **kwargs:
            See :class:`pytorch_forecasting.models.base_model.AutoRegressiveBaseModel`.
    """

    def __init__(
        self,
        target: Union[str, Sequence[str]],
        target_lags: Dict[str, Dict[str, int]],  # pylint: disable=unused-argument
        n_layers: int,
        hidden_size: int,
        dropout: float = 0.1,
        input_size: Optional[int] = None,
        loss: Optional[Metric] = None,
        **kwargs: Any,
    ):
        """Prefer using the `LSTMModel.from_dataset()` method rather than this constructor.

        Args:
            target (Union[str, Sequence[str]]):
                Name (or list of names) of target variable(s).
            target_lags (Dict[str, Dict[str, int]]): _description_

            n_layers (int):
                Number of LSTM layers.

            hidden_size (int):
                Hidden size for LSTM model.

            dropout (float, optional):
                Droput probability (<1). Defaults to 0.1.

            input_size (int, optional):
                Input size. Defaults to: inferred from `target`.

            loss (Metric):
                Loss criterion. Can be different for each target in multi-target setting thanks to
                `MultiLoss`. Defaults to `MAE`.

            **kwargs:
                See :class:`pytorch_forecasting.models.base_model.AutoRegressiveBaseModel`.
        """
        n_targets = len(target) if isinstance(target, (list, tuple)) else 1
        if input_size is None:
            input_size = n_targets
        # arguments target and target_lags are required for autoregressive models
        # even though target_lags cannot be used without covariates
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # loss
        if loss is None:
            loss = MultiLoss([MAE() for _ in range(n_targets)]) if n_targets > 1 else MAE()  # type: ignore
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(loss=loss, **kwargs)  # type: ignore
        # use version of LSTM that can handle zero-length sequences
        self.lstm = LSTM(
            hidden_size=hidden_size,
            input_size=input_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        # output layer
        self.output_layer = nn.Linear(hidden_size, n_targets)
        # others
        self._input_vector: Tensor

    def encode(self, x: Dict[str, torch.Tensor]) -> Tuple[Tensor, Tensor]:
        """Encode method.
        Args:
            x (Dict[str, torch.Tensor]):
                First item returned by a `DataLoader` obtained from `TimeSeriesDataset.to_dataloader()`.
        Returns:
            Tuple[Tensor, Tensor]:
                Tuple of hidden and cell state.
        """
        # we need at least one encoding step as because the target needs to be lagged by one time step
        # because we use the custom LSTM, we do not have to require encoder lengths of > 1
        # but can handle lengths of >= 1
        encoder_lengths = x["encoder_lengths"]
        assert encoder_lengths.min() >= 1, f"encoder_lengths = {encoder_lengths.min()}"
        input_vector = x["encoder_cont"].clone()
        # lag target by one
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions],
            shifts=1,
            dims=1,
        )
        input_vector = input_vector[:, 1:]  # first time step cannot be used because of lagging
        # determine effective encoder_length length
        effective_encoder_lengths = x["encoder_lengths"] - 1
        # run through LSTM network
        hidden_state: Tuple[Tensor, Tensor]
        _, hidden_state = self.lstm(
            input_vector,
            lengths=effective_encoder_lengths,
            enforce_sorted=False,  # passing the lengths directly
        )  # second ouput is not needed (hidden state)
        return hidden_state

    def decode(
        self,
        x: Dict[str, torch.Tensor],
        hidden_state: Tuple[Tensor, Tensor],
    ) -> Union[List[Tensor], Tensor]:
        """
        Args:
            x (Dict[str, torch.Tensor]):
                First item returned by a `DataLoader` obtained from `TimeSeriesDataset.to_dataloader()`.
            hidden_state (Tuple[Tensor, Tensor]):
                Tuple of hidden and cell state.
        Returns:
            (Union[List[Tensor], Tensor]):
                Tensor if one target, list of Tensors if multi-target.
        """
        # again lag target by one
        input_vector = x["decoder_cont"].clone()  # (B,L,D)
        B, L, D = input_vector.size()
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )
        # but this time fill in missing target from encoder_cont at the first time step instead of throwing it away
        last_encoder_target = x["encoder_cont"][
            torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
            x["encoder_lengths"] - 1,
            self.target_positions.unsqueeze(-1),
        ].T
        input_vector[:, 0, self.target_positions] = last_encoder_target
        # Training mode
        if self.training:  # training mode
            lstm_output, _ = self.lstm(input_vector, hidden_state, lengths=x["decoder_lengths"], enforce_sorted=False)
            # transform into right shape
            out: Tensor = self.output_layer(lstm_output)
            if self.n_targets > 1:
                out = [out[:, :, i].view(B, L, -1) for i in range(D)]  # type: ignore
            prediction: List[Tensor] = self.transform_output(out, target_scale=x["target_scale"])
            # predictions are not yet rescaled
            return prediction
        # Prediction mode
        self._input_vector = input_vector
        n_decoder_steps = input_vector.size(1)
        first_target = input_vector[:, 0, :]  # self.target_positions?
        first_target = first_target.view(B, 1, D)
        target_scale = x["target_scale"]
        output: Union[List[Tensor], Tensor] = self.decode_autoregressive(
            self.decode_one,  # make predictions which are fed into next step
            first_target=first_target,
            first_hidden_state=hidden_state,
            target_scale=target_scale,
            n_decoder_steps=n_decoder_steps,
        )
        # predictions are already rescaled
        return output

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        Args:
            x (Dict[str, torch.Tensor]): Input dict.

        Returns:
            Dict[str, torch.Tensor]: Output dict.
        """
        hidden_state = self.encode(x)  # encode to hidden state
        output = self.decode(x, hidden_state)  # decode leveraging hidden state
        out = self.to_network_output(prediction=output)
        return out

    def decode_one(
        self,
        idx: int,
        lagged_targets: List[Tensor],
        hidden_state: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """_summary_
        Args:
            idx (int):
                (???).
            lagged_targets (List[Tensor]):
                (???).
            hidden_state (Tuple[Tensor, Tensor]):
                `(h,c)` (hidden state, cell state).
        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]:
                One-step-ahead prediction and tuple of `(h,c)` (hidden state, cell state).
        """
        B, _, D = self._input_vector.size()

        # input has shape (B,L,D)
        x = self._input_vector[:, [idx]]
        # take most recent target (i.e. lag=1)
        lag = lagged_targets[-1]
        assert lag.size(0) == B
        assert lag.size(-1) == D
        # make sure it has shape (B,D)
        lag = lag.view(B, D)
        # overwrite at target positions
        x[:, 0, :] = lag
        lstm_output, hidden_state = self.lstm(x, hidden_state)
        # transform into right shape
        prediction: Tensor = self.output_layer(lstm_output)[:, 0]  # take first timestep
        if self.n_targets > 1:
            prediction = prediction.view(B, 1, D)
        return prediction, hidden_state
