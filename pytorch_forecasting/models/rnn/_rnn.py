"""
Simple recurrent model - either with LSTM or GRU cells.
"""

from copy import copy
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from pytorch_forecasting.data.encoders import MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    MASE,
    RMSE,
    SMAPE,
    MultiHorizonMetric,
    MultiLoss,
    QuantileLoss,
)
from pytorch_forecasting.models.base import AutoRegressiveBaseModelWithCovariates
from pytorch_forecasting.models.nn import HiddenState, MultiEmbedding, get_rnn
from pytorch_forecasting.utils import apply_to_list, to_list


class RecurrentNetwork(AutoRegressiveBaseModelWithCovariates):
    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.rnn._rnn_pkg import (
            RecurrentNetwork_pkg,
        )

        return RecurrentNetwork_pkg

    def __init__(
        self,
        cell_type: str = "LSTM",
        hidden_size: int = 10,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        static_categoricals: list[str] | None = None,
        static_reals: list[str] | None = None,
        time_varying_categoricals_encoder: list[str] | None = None,
        time_varying_categoricals_decoder: list[str] | None = None,
        categorical_groups: dict[str, list[str]] | None = None,
        time_varying_reals_encoder: list[str] | None = None,
        time_varying_reals_decoder: list[str] | None = None,
        embedding_sizes: dict[str, tuple[int, int]] | None = None,
        embedding_paddings: list[str] | None = None,
        embedding_labels: dict[str, np.ndarray] | None = None,
        x_reals: list[str] | None = None,
        x_categoricals: list[str] | None = None,
        output_size: int | list[int] = 1,
        target: str | list[str] = None,
        target_lags: dict[str, list[int]] | None = None,
        loss: MultiHorizonMetric = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """Recurrent network.

        Simple LSTM/GRU layer followed by an output projection.

        Parameters
        ----------
        cell_type : str, default="LSTM"
            Recurrent cell type. One of {"LSTM", "GRU"}.
        hidden_size : int, default=10
            Hidden state size of the recurrent layer.
        rnn_layers : int, default=2
            Number of recurrent layers.
        dropout : float, default=0.1
            Dropout in recurrent layers (only applied when ``rnn_layers > 1``).
        static_categoricals : list[str] or None, default=None
            Names of static categorical variables.
        static_reals : list[str] or None, default=None
            Names of static continuous variables.
        time_varying_categoricals_encoder : list[str] or None, default=None
            Names of categorical variables used in the encoder.
        time_varying_categoricals_decoder : list[str] or None, default=None
            Names of categorical variables used in the decoder.
        categorical_groups : dict[str, list[str]] or None, default=None
            Mapping from a grouped categorical name to the list of categorical variables that form it.
        time_varying_reals_encoder : list[str] or None, default=None
            Names of continuous variables used in the encoder.
        time_varying_reals_decoder : list[str] or None, default=None
            Names of continuous variables used in the decoder.
        embedding_sizes : dict[str, tuple[int, int]] or None, default=None
            Mapping from categorical variable name to ``(n_classes, embedding_dim)``.
        embedding_paddings : list[str] or None, default=None
            Names of embeddings for which the ``0`` index is treated as padding.
        embedding_labels : dict[str, numpy.ndarray] or None, default=None
            Mapping from categorical variable name to the array of category labels.
        x_reals : list[str] or None, default=None
            Order of continuous variables in the tensor passed to :meth:`forward`.
        x_categoricals : list[str] or None, default=None
            Order of categorical variables in the tensor passed to :meth:`forward`.
        output_size : int or list[int], default=1
            Number of outputs per time step. For multi-target models, pass a list with one output size per target.
        target : str or list[str], default=None
            Target variable name or list of target variable names.
        target_lags : dict[str, list[int]] or None, default=None
            Mapping from target name to a list of lags (time steps) to include as additional inputs.
        loss : MultiHorizonMetric, optional
            Loss function taking prediction and targets.
        logging_metrics : torch.nn.ModuleList, optional
            Metrics to log during training. Defaults to
            ``nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])``.
        **kwargs
            Additional arguments passed to the base model.
        """
        if static_categoricals is None:
            static_categoricals = []
        if static_reals is None:
            static_reals = []
        if time_varying_categoricals_encoder is None:
            time_varying_categoricals_encoder = []
        if time_varying_categoricals_decoder is None:
            time_varying_categoricals_decoder = []
        if categorical_groups is None:
            categorical_groups = {}
        if time_varying_reals_encoder is None:
            time_varying_reals_encoder = []
        if time_varying_reals_decoder is None:
            time_varying_reals_decoder = []
        if embedding_sizes is None:
            embedding_sizes = {}
        if embedding_paddings is None:
            embedding_paddings = []
        if embedding_labels is None:
            embedding_labels = {}
        if x_reals is None:
            x_reals = []
        if x_categoricals is None:
            x_categoricals = []
        if target_lags is None:
            target_lags = {}
        if loss is None:
            loss = MAE()
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        self.save_hyperparameters()
        # store loss function separately as it is a module
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        self.embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=embedding_paddings,
            categorical_groups=categorical_groups,
            x_categoricals=x_categoricals,
        )

        lagged_target_names = [l for lags in target_lags.values() for l in lags]
        assert set(self.encoder_variables) - set(to_list(target)) - set(
            lagged_target_names
        ) == set(self.decoder_variables) - set(lagged_target_names), (
            "Encoder and decoder variables have to"
            " be the same apart from target variable"
        )
        for targeti in to_list(target):
            assert (
                targeti in time_varying_reals_encoder
            ), f"target {targeti} has to be real"  # todo: remove this restriction
        assert (isinstance(target, str) and isinstance(loss, MultiHorizonMetric)) or (
            isinstance(target, tuple | list)
            and isinstance(loss, MultiLoss)
            and len(loss) == len(target)
        ), "number of targets should be equivalent to number of loss metrics"

        rnn_class = get_rnn(cell_type)
        cont_size = len(self.reals)
        cat_size = sum(self.embeddings.output_size.values())
        input_size = cont_size + cat_size
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.rnn_layers,
            dropout=self.hparams.dropout if self.hparams.rnn_layers > 1 else 0,
            batch_first=True,
        )

        # add linear layers for argument projects
        if isinstance(target, str):  # single target
            self.output_projector = nn.Linear(
                self.hparams.hidden_size, self.hparams.output_size
            )
            assert not isinstance(
                self.loss, QuantileLoss
            ), "QuantileLoss does not work with recurrent network"
        else:  # multi target
            self.output_projector = nn.ModuleList(
                [
                    nn.Linear(self.hparams.hidden_size, size)
                    for size in self.hparams.output_size
                ]
            )
            for l in self.loss:
                assert not isinstance(
                    l, QuantileLoss
                ), "QuantileLoss does not work with recurrent network"

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: list[str] = None,
        **kwargs,
    ):
        """Create model from a dataset.

        Parameters
        ----------
        dataset : TimeSeriesDataSet
            Time series dataset.
        allowed_encoder_known_variable_names : list[str] or None, default=None
            List of known variables that are allowed in the encoder. Defaults to all.
        **kwargs
            Additional arguments such as hyperparameters for the model (see :meth:`__init__`).

        Returns
        -------
        RecurrentNetwork
            Initialized model.
        """
        new_kwargs = copy(kwargs)
        new_kwargs.update(
            cls.deduce_default_output_parameters(
                dataset=dataset, kwargs=kwargs, default_loss=MAE()
            )
        )
        assert (
            not isinstance(dataset.target_normalizer, NaNLabelEncoder)
            and (
                not isinstance(dataset.target_normalizer, MultiNormalizer)
                or all(
                    not isinstance(normalizer, NaNLabelEncoder)
                    for normalizer in dataset.target_normalizer
                )
            )
        ), (
            "target(s) should be continuous - categorical targets are not supported"
        )  # todo: remove this restriction # noqa: E501
        return super().from_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            **new_kwargs,
        )

    def construct_input_vector(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        one_off_target: torch.Tensor = None,
    ) -> torch.Tensor:
        """Create input vector for the RNN.

        The target is shifted by one time step. During decoding, the first shifted target value can be provided via
        ``one_off_target``.

        Parameters
        ----------
        x_cat : torch.Tensor
            Categorical inputs.
        x_cont : torch.Tensor
            Continuous inputs.
        one_off_target : torch.Tensor or None, default=None
            Target values to insert into the first position after shifting. If None, the first time step is removed.

        Returns
        -------
        torch.Tensor
            Input vector that is fed into the RNN.
        """
        # create input vector
        if len(self.categoricals) > 0:
            embeddings = self.embeddings(x_cat)
            flat_embeddings = torch.cat(list(embeddings.values()), dim=-1)
            input_vector = flat_embeddings

        if len(self.reals) > 0:
            input_vector = x_cont.clone()

        if len(self.reals) > 0 and len(self.categoricals) > 0:
            input_vector = torch.cat([x_cont, flat_embeddings], dim=-1)

        # shift target by one
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )

        if one_off_target is not None:  # set first target input (which is rolled over)
            input_vector[:, 0, self.target_positions] = one_off_target
        else:
            input_vector = input_vector[:, 1:]

        # shift target
        return input_vector

    def encode(self, x: dict[str, torch.Tensor]) -> HiddenState:
        """Encode the encoder sequence into a hidden state.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Network input batch as produced by :class:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.

        Returns
        -------
        HiddenState
            Hidden state for initializing the decoder.
        """
        # encode using rnn
        assert x["encoder_lengths"].min() > 0
        encoder_lengths = x["encoder_lengths"] - 1
        input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])
        _, hidden_state = self.rnn(
            input_vector, lengths=encoder_lengths, enforce_sorted=False
        )  # second output is not needed (hidden state)
        return hidden_state

    def decode_all(
        self,
        x: torch.Tensor,
        hidden_state: HiddenState,
        lengths: torch.Tensor = None,
    ):
        decoder_output, hidden_state = self.rnn(
            x, hidden_state, lengths=lengths, enforce_sorted=False
        )
        if isinstance(self.hparams.target, str):  # single target
            output = self.output_projector(decoder_output)
        else:
            output = [projector(decoder_output) for projector in self.output_projector]
        return output, hidden_state

    def decode(
        self,
        input_vector: torch.Tensor,
        target_scale: torch.Tensor,
        decoder_lengths: torch.Tensor,
        hidden_state: HiddenState,
        n_samples: int = None,
    ) -> tuple[torch.Tensor, bool]:
        """Decode the hidden state into a prediction.

        During training, decoding uses teacher forcing. During evaluation, decoding is autoregressive, i.e. past
        predictions are fed back as inputs.

        Parameters
        ----------
        input_vector : torch.Tensor
            Decoder input vector.
        target_scale : torch.Tensor
            Target scale used for rescaling outputs.
        decoder_lengths : torch.Tensor
            Lengths of decoder sequences.
        hidden_state : HiddenState
            Hidden state returned by :meth:`encode`.
        n_samples : int or None, optional
            Reserved for API compatibility. Currently unused for this model.

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            Model predictions.
        """
        if self.training:
            output, _ = self.decode_all(
                input_vector, hidden_state, lengths=decoder_lengths
            )
            output = self.transform_output(output, target_scale=target_scale)
        else:
            # run in eval, i.e. simulation mode
            target_pos = self.target_positions
            lagged_target_positions = self.lagged_target_positions

            # define function to run at every decoding step
            def decode_one(
                idx,
                lagged_targets,
                hidden_state,
            ):
                x = input_vector[:, [idx]]
                x[:, 0, target_pos] = lagged_targets[-1]
                for lag, lag_positions in lagged_target_positions.items():
                    if idx > lag:
                        x[:, 0, lag_positions] = lagged_targets[-lag]
                prediction, hidden_state = self.decode_all(x, hidden_state)
                prediction = apply_to_list(
                    prediction, lambda x: x[:, 0]
                )  # select first time step
                return prediction, hidden_state

            # make predictions which are fed into next step
            output = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],
                first_hidden_state=hidden_state,
                target_scale=target_scale,
                n_decoder_steps=input_vector.size(1),
            )
        return output

    def forward(
        self, x: dict[str, torch.Tensor], n_samples: int = None
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Network input batch as produced by :class:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.
        n_samples : int or None, optional
            Reserved for API compatibility. Currently unused for this model.

        Returns
        -------
        dict[str, torch.Tensor]
            Network output dictionary containing the prediction.
        """
        hidden_state = self.encode(x)
        # decode
        input_vector = self.construct_input_vector(
            x["decoder_cat"],
            x["decoder_cont"],
            one_off_target=x["encoder_cont"][
                torch.arange(
                    x["encoder_cont"].size(0), device=x["encoder_cont"].device
                ),
                x["encoder_lengths"] - 1,
                self.target_positions.unsqueeze(-1),
            ].T.contiguous(),
        )

        output = self.decode(
            input_vector,
            decoder_lengths=x["decoder_lengths"],
            target_scale=x["target_scale"],
            hidden_state=hidden_state,
        )
        # return relevant part
        return self.to_network_output(prediction=output)
