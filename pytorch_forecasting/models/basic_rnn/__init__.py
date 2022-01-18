"""
Basic RNN model with LSTM or GRU cells
"""
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn.utils import rnn

from pytorch_forecasting.metrics import MultiHorizonMetric
from pytorch_forecasting.models.base_model import AutoRegressiveBaseModelWithCovariates
from pytorch_forecasting.models.nn import MultiEmbedding, get_rnn
from pytorch_forecasting.utils import to_list


class LSTMModel(AutoRegressiveBaseModelWithCovariates):
    """
    Basic RNN network.
    """

    def __init__(
        self,
        cell_type: str = "LSTM",
        hidden_size: int = 10,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        n_validation_samples: int = None,
        n_plotting_samples: int = None,
        target: Union[str, List[str]] = None,
        loss: MultiHorizonMetric = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """
        Args:
            cell_type (str, optional): Recurrent cell type ["LSTM", "GRU"]. Defaults to "LSTM".
            hidden_size (int, optional): hidden recurrent size - the most important hyperparameter along with
                ``rnn_layers``. Defaults to 10.
            rnn_layers (int, optional): Number of RNN layers - important hyperparameter. Defaults to 2.
            dropout (float, optional): Dropout in RNN layers. Defaults to 0.1.
            static_categoricals: integer of positions of static categorical variables
            static_reals: integer of positions of static continuous variables
            time_varying_categoricals_encoder: integer of positions of categorical variables for encoder
            time_varying_categoricals_decoder: integer of positions of categorical variables for decoder
            time_varying_reals_encoder: integer of positions of continuous variables for encoder
            time_varying_reals_decoder: integer of positions of continuous variables for decoder
            categorical_groups: dictionary where values
                are list of categorical variables that are forming together a new categorical
                variable which is the key in the dictionary
            x_reals: order of continuous variables in tensor passed to forward function
            x_categoricals: order of categorical variables in tensor passed to forward function
            embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
                embedding size
            embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
            embedding_labels: dictionary mapping (string) indices to list of categorical labels
            n_validation_samples (int, optional): Number of samples to use for calculating validation metrics.
                Defaults to None, i.e. no sampling at validation stage and using "mean" of distribution for logging
                metrics calculation.
            n_plotting_samples (int, optional): Number of samples to generate for plotting predictions
                during training. Defaults to ``n_validation_samples`` if not None or 100 otherwise.
            target (str, optional): Target variable or list of target variables. Defaults to None.
            loss (DistributionLoss, optional): Distribution loss function. Keep in mind that each distribution
                loss function might have specific requirements for target normalization.
                Defaults to :py:class:`~pytorch_forecasting.metrics.NormalDistributionLoss`.
            logging_metrics (nn.ModuleList, optional): Metrics to log during training.
                Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()]).
        """
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        assert set(self.encoder_variables) - set(to_list(target)) == set(
            self.decoder_variables
        ), "Encoder and decoder variables have to be the same apart from target variable"
        for targeti in to_list(target):
            assert (
                targeti in time_varying_reals_encoder
            ), f"target {targeti} has to be real"  # todo: remove this restriction

        self.embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=embedding_paddings,
            categorical_groups=categorical_groups,
            x_categoricals=x_categoricals,
        )

        time_series_rnn = get_rnn(cell_type)
        cont_size = len(self.reals)
        cat_size = sum(self.embeddings.output_size.values())
        input_size = cont_size + cat_size
        self.rnn = time_series_rnn(
            input_size=input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.rnn_layers,
            dropout=self.hparams.dropout if self.hparams.rnn_layers > 1 else 0,
            batch_first=True,
        )

        # add linear layers for argument projects
        if isinstance(target, str):  # single target
            self.output_projector = nn.Linear(self.hparams.hidden_size, 1)
        else:  # multi target
            self.output_projector = nn.ModuleList([nn.Linear(self.hparams.hidden_size, 1) for _ in target])

    @property
    def target_position(self):
        # position of target within reals vector: with covariates: self.hparams.x_reals.index(self.hparams.target)
        return 0

    def encode(self, x: Dict[str, torch.Tensor]):
        # we need at least one encoding step as because the target needs to be lagged by one time step
        # as we are lazy, we also require that the encoder length is at least 1, so we can easily generate a
        # hidden state here. See the DeepAR implementation for how to use a minimal encoder length of 1
        max_encoder_length = x["encoder_lengths"].max()
        assert x["encoder_lengths"].min() > 0
        if max_encoder_length > 1:
            encoder_lengths = x["encoder_lengths"] - 1
            rnn_encoder_lengths = encoder_lengths.where(encoder_lengths > 0, torch.ones_like(encoder_lengths))
            input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])
            _, hidden_state = self.rnn(
                rnn.pack_padded_sequence(
                    input_vector, rnn_encoder_lengths.cpu(), enforce_sorted=False, batch_first=True
                )
            )  # second ouput is not needed (hidden state)
            # replace hidden cell with initial input if encoder_length is zero to determine correct initial state
            no_encoding = (encoder_lengths == 0)[None, :, None]  # shape: n_lstm_layers x batch_size x hidden_size
            hidden_state = self.rnn.handle_no_encoding(hidden_state, no_encoding)
        else:
            hidden_state = self.rnn.init_hidden_state(x, self.hparam.hidden_size)
        return hidden_state

    def decode(self, x: Dict[str, torch.Tensor], hidden_state):
        # again lag target by one
        input_vector = x["decoder_cont"].clone()
        input_vector[..., self.target_position] = torch.roll(input_vector[..., self.target_position], shifts=1, dims=1)
        # but this time fill in missing target from encoder_cont at the first time step instead of throwing it away
        last_encoder_target = x["encoder_cont"][
            torch.arange(x["encoder_cont"].size(0)), x["encoder_lengths"] - 1, self.target_position
        ]
        input_vector[:, 0, self.target_position] = last_encoder_target

        if self.training:  # training attribute is provided from PyTorch and indicates if module is in training model
            packed_decoder = rnn.pack_padded_sequence(
                input_vector, lengths=x["decoder_lengths"].cpu(), batch_first=True, enforce_sorted=False
            )
            # run through same lstm
            lstm_output, _ = self.lstm(packed_decoder, hidden_state)
            # unpack sequence
            lstm_output, _ = rnn.pad_packed_sequence(lstm_output, batch_first=True)
            # transform into right shape
            prediction = self.output_layer(lstm_output)

        else:  # if not training, need to predict in autoregressive manner
            # predict one by one
            max_decoder_length = x["decoder_lengths"].max()
            # initialize previous target and hidden state
            last_target = last_encoder_target
            last_hidden_state = hidden_state

            predictions = []

            # for each time step run prediction
            for i in range(max_decoder_length):
                current_input_vector = input_vector[:, i].unsqueeze(1)  # select time step in decoder
                current_input_vector[:, 0, self.target_position] = last_target  # insert previous target

                # make lstm prediction
                lstm_prediction, new_hidden_state = self.lstm(current_input_vector, last_hidden_state)
                prediction = self.output_layer(lstm_prediction).squeeze(1)

                # save prediction
                predictions.append(prediction)

                # prepare for next time step
                last_hidden_state = new_hidden_state

                # Prediction should be passed through transformer and then inversely transformed.
                # The inverse transformation might be only approximately the inverse of the
                # forward transformation making this step important.
                rescaled_prediction = self.transform_output(
                    prediction=prediction, target_scale=x["target_scale"]
                )  # inverse transform
                normalized_prediction = self.output_transformer.transform(
                    rescaled_prediction, target_scale=x["target_scale"]
                )  # transform

                last_target = normalized_prediction.squeeze(1)

            # stack all predictions
            prediction = torch.stack(predictions, dim=1)

        return prediction

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hidden_state = self.encode(x)  # encode to hidden state
        prediction = self.decode(x, hidden_state)  # decode leveraging hidden state
        return self.to_network_output(prediction=prediction)
