"""
`DeepAR: Probabilistic forecasting with autoregressive recurrent networks`
<https://www.sciencedirect.com/science/article/pii/S0169207019301888>`_
which is the one of the most popular forecasting algorithms and is often used as a baseline
"""
from typing import Dict, List, Tuple, Union

import numpy as np
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.distributions as dists
import torch.nn as nn
from torch.nn.utils import rnn

from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, DistributionLoss, NormalDistributionLoss
from pytorch_forecasting.models.base_model import AutoRegressiveBaseModelWithCovariates
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding


class DeepAR(AutoRegressiveBaseModelWithCovariates):
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
        target: str = None,
        loss: DistributionLoss = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        if loss is None:
            loss = NormalDistributionLoss()
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

        assert set(self.encoder_variables) - {target} == set(
            self.decoder_variables
        ), "Encoder and decoder variables have to be the same apart from target variable"
        assert target in time_varying_reals_encoder, "target has to be real"  # todo: remove this restriction

        rnn = getattr(nn, cell_type)
        self.rnn = rnn(
            input_size=self.input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.rnn_layers,
            dropout=self.hparams.dropout if self.hparams.rnn_layers > 1 else 0,
            batch_first=True,
        )

        # add linear layers for argument projects
        self.distribution_projector = nn.Linear(self.hparams.hidden_size, len(self.loss.distribution_arguments))

    @property
    def input_size(self):
        cont_size = len(self.reals)
        cat_size = sum([size[1] for size in self.hparams.embedding_sizes.values()])
        return cont_size + cat_size

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: List[str] = None,
        **kwargs,
    ):
        """
        Create model from dataset.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            DeepAR
        """
        # assert fixed encoder and decoder length for the moment
        new_kwargs = {}
        new_kwargs.update(kwargs)
        return super().from_dataset(
            dataset, allowed_encoder_known_variable_names=allowed_encoder_known_variable_names, **new_kwargs
        )

    def construct_input_vector(
        self, x_cat: torch.Tensor, x_cont: torch.Tensor, one_off_target: torch.Tensor = None
    ) -> torch.Tensor:
        """
        input dimensions: n_samples x time x variables
        """
        # create input vector
        if len(self.categoricals) > 0:
            embeddings = self.embeddings(x_cat)
            flat_embeddings = torch.cat([emb for emb in embeddings.values()], dim=-1)
            input_vector = flat_embeddings

        if len(self.reals) > 0:
            input_vector = x_cont

        if len(self.reals) > 0 and len(self.categoricals) > 0:
            input_vector = torch.cat([x_cont, flat_embeddings], dim=-1)

        # shift target by one
        input_vector[..., self.target_position] = torch.roll(input_vector[..., self.target_position], shifts=1, dims=1)

        if one_off_target is not None:  # set first target input (which is rolled over)
            input_vector[:, 0, self.target_position] = one_off_target
        else:
            input_vector = input_vector[:, 1:]

        # shift target
        return input_vector

    @property
    def target_position(self):
        variables = self.hparams.time_varying_reals_encoder  # todo: support categorical targets
        pos = variables.index(self.hparams.target)
        return pos

    def encode(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode using rnn
        max_encoder_length = x["encoder_lengths"].max()
        assert x["encoder_lengths"].min() > 0
        if max_encoder_length > 1:
            encoder_lengths = x["encoder_lengths"] - 1
            rnn_encoder_lengths = encoder_lengths.where(encoder_lengths > 0, torch.ones_like(encoder_lengths))
            input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])
            _, (hidden, cell) = self.rnn(
                rnn.pack_padded_sequence(input_vector, rnn_encoder_lengths, enforce_sorted=False, batch_first=True)
            )  # second ouput is not needed (hidden state)
            # replace hidden cell with initial input if encoder_length is zero to determine correct initial state
            no_encoding = (encoder_lengths == 0)[None, :, None]  # shape: n_lstm_layers x batch_size x hidden_size
            hidden = hidden.masked_fill(no_encoding, 0.0)
            cell = cell.masked_fill(no_encoding, 0.0)
        else:
            hidden = torch.zeros(
                (x["encoder_cont"].size(0), self.hparams.hidden_size),
                device=x["decoder_cont"].device,
                dtype=torch.float,
            )
            cell = torch.zeros(
                (x["encoder_cont"].size(0), self.hparams.hidden_size),
                device=x["decoder_cont"].device,
                dtype=torch.float,
            )
        return hidden, cell

    def decode(self, input_vector, hidden: torch.Tensor, cell: torch.Tensor, train: bool = False) -> torch.Tensor:

        if train:
            decoder_output, _ = self.rnn(
                input_vector,
                (hidden, cell),
            )
            decoder_output, _ = rnn.pad_packed_sequence(decoder_output, batch_first=True)
            output = self.distribution_projector(decoder_output)

        else:
            # run in eval, i.e. simulation mode
            target_pos = self.target_position
            input_target = input_vector[:, 0, target_pos]
            output = []
            for idx in range(input_vector.size(1)):
                x = input_vector[:, [idx]]
                x[:, 0, target_pos] = input_target
                decoder_output, (hidden, cell) = self.rnn(x, (hidden, cell))
                input_target = self.distribution_projector(decoder_output)
                # transform into real space
                # todo: this needs to be revised: sample in real space
                #  easier to understand intuitively (e.g. Possion) but
                #  can lead to strange outcomes (e.g. strictly positive gaussian?)

                input_target = self.transform_output(input_target)
                # sample value(s) from distribution
                input_target = self.loss.sample(input_target)
                # normalize prediction prediction
                input_target = self.output_transformer.transform(
                    dict(input_target, groups=x["groups"], target_scale=x["target_scale"])
                )
                output.append(input_target)
            output = torch.cat(output, dim=1)

        return output

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        hidden, cell = self.encode(x)
        # decode
        input_vector = self.construct_input_vector(
            x["decoder_cat"], x["decoder_cont"], one_off_target=x["encoder_cont"][:, -1, self.target_position]
        )
        input_vector = rnn.pack_padded_sequence(
            input_vector, x["decoder_lengths"], enforce_sorted=False, batch_first=True
        )
        output = self.decode(input_vector, hidden, cell, train=True)
        # return relevant part
        return dict(
            prediction=output,
            groups=x["groups"],
            decoder_time_idx=x["decoder_time_idx"],
            target_scale=x["target_scale"],
        )

    def transform_output(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return out
        elif self.output_transformer is None:
            out = out["prediction"]
        else:
            out = self.output_transformer(out)  # todo: needs to be revised - probably depending on loss
        return out
