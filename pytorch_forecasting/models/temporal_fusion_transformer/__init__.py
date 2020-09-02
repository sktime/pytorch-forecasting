"""
The temporal fusion transformer is a powerful predictive model for forecasting timeseries
"""
from typing import Callable, Union, List, Dict, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.utils import rnn

from pytorch_forecasting.models.base_model import BaseModel, CovariatesMixin
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import MultiHorizonMetric, QuantileLoss, SMAPE, MAE, RMSE, MAPE
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    VariableSelectionNetwork,
    GatedResidualNetwork,
    GateAddNorm,
    InterpretableMultiHeadAttention,
    GatedLinearUnit,
    AddNorm,
    TimeDistributedEmbeddingBag,
)
from pytorch_forecasting.utils import autocorrelation, integer_histogram, get_embedding_size


class TemporalFusionTransformer(BaseModel, CovariatesMixin):
    def __init__(
        self,
        hidden_size: int = 16,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        output_size: int = 7,
        loss: MultiHorizonMetric = QuantileLoss(),
        attention_head_size: int = 4,
        max_encoder_length: int = 10,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        hidden_continuous_size: int = 8,
        hidden_continuous_sizes: Dict[str, int] = {},
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        learning_rate: float = 1e-3,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        log_gradient_flow: bool = False,
        reduce_on_plateau_patience: int = 1000,
        monotone_constaints: Dict[str, int] = {},
        share_single_variable_networks: bool = False,
        output_transformer: Callable = None,
    ):
        """
        Temporal Fusion Transformer for forecasting timeseries - use its :py:meth:`~from_dataset` method if possible.

        Args:

            hidden_size: hidden size of network which is its main hyperparameter and can range from 8 to 512
            lstm_layers: number of LSTM layers (2 is mostly optimal)
            dropout: dropout rate
            output_size: number of outputs (e.g. number of quantiles for QuantileLoss)
            loss: loss function taking prediction and targets
            attention_head_size: number of attention heads (4 is a good default)
            max_encoder_length: length to encode (can be far longer than the decoder length but does not have to be)
            static_categoricals: integer of positions of static categorical variables
            static_reals: integer of positions of static continuous variables
            time_varying_categoricals_encoder: integer of positions of categorical variables for encoder
            time_varying_categoricals_decoder: integer of positions of categorical variables for decoder
            time_varying_reals_encoder: integer of positions of continuous variables for encoder
            time_varying_reals_decoder: integer of positions of continuous variables for decoder
            x_reals: order of continuous variables in tensor passed to forward function
            x_categoricals: order of categorical variables in tensor passed to forward function
            hidden_continuous_size: default for hidden size for processing continous variables (similar to categorical
                embedding size)
            hidden_continuous_sizes: dictionary mapping continuous input indices to sizes for variable selection
                (fallback to hidden_continuous_size if index is not in dictionary)
            embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
                embedding size
            embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
            embedding_labels: dictionary mapping (string) indices to list of categorical labels
            learning_rate: learning rate
            log_interval: log predictions every x batches, do not log if 0 or less, log interpretation if > 0. If < 1.0
                , will log multiple entries per batch. Defaults to -1.
            log_val_interval: frequency with which to log validation set metrics, defaults to log_interval
            log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
                failures
            reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10
            monotone_constaints (Dict[str, int]): dictionary of monotonicity constraints for continuous decoder
                variables mapping
                position (e.g. ``"0"`` for first position) to constraint (``-1`` for negative and ``+1`` for positive,
                larger numbers add more weight to the constraint vs. the loss but are usually not necessary).
                This constraint significantly slows down training. Defaults to {}.
            share_single_variable_networks (bool): if to share the single variable networks between the encoder and
                decoder. Defaults to False.
        """
        self.save_hyperparameters()
        super().__init__()
        # store loss function separately as it is a module
        assert isinstance(loss, MultiHorizonMetric), "Loss has to of class `MultiHorizonMetric`"
        self.loss = loss
        self.output_transformer = output_transformer
        self.logging_metrics = [SMAPE(), MAE(), RMSE(), MAPE()]

        # processing inputs
        # embeddings
        self.input_embeddings = nn.ModuleDict()
        for name in self.hparams.embedding_sizes.keys():
            embedding_size = min(self.hparams.embedding_sizes[name][1], self.hparams.hidden_size)
            # convert to list to become mutable
            self.hparams.embedding_sizes[name] = list(self.hparams.embedding_sizes[name])
            self.hparams.embedding_sizes[name][1] = embedding_size
            if name in self.hparams.categorical_groups:  # embedding bag if related embeddings
                self.input_embeddings[name] = TimeDistributedEmbeddingBag(
                    self.hparams.embedding_sizes[name][0], embedding_size, mode="sum", batch_first=True
                )
            else:
                if name in self.hparams.embedding_paddings:
                    padding_idx = 0
                else:
                    padding_idx = None
                self.input_embeddings[name] = nn.Embedding(
                    self.hparams.embedding_sizes[name][0],
                    embedding_size,
                    padding_idx=padding_idx,
                )

        # continuous variable processing
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(1, self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size))
                for name in self.hparams.x_reals
            }
        )

        # variable selection
        # variable selection for static variables
        static_input_sizes = {name: self.hparams.embedding_sizes[name][1] for name in self.hparams.static_categoricals}
        static_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.static_reals
            }
        )
        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.static_categoricals},
            dropout=self.hparams.dropout,
            prescalers=self.prescalers,
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.hparams.embedding_sizes[name][1] for name in self.hparams.time_varying_categoricals_encoder
        }
        encoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.time_varying_reals_encoder
            }
        )

        decoder_input_sizes = {
            name: self.hparams.embedding_sizes[name][1] for name in self.hparams.time_varying_categoricals_decoder
        }
        decoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(name, self.hparams.hidden_continuous_size)
                for name in self.hparams.time_varying_reals_decoder
            }
        )

        # create single variable grns that are shared across decoder and encoder

        if self.hparams.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hparams.hidden_size),
                    self.hparams.hidden_size,
                    self.hparams.dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, self.hparams.hidden_size),
                        self.hparams.hidden_size,
                        self.hparams.dropout,
                    )

        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_encoder},
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={name: True for name in self.hparams.time_varying_categoricals_decoder},
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        # static encoders
        # for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # for hidden state of the lstm
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # for cell state of the lstm
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.dropout
        )

        # lstm encoder (history) and decoder (future) for local processing
        self.lstm_encoder = nn.LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = nn.LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # skip connection for lstm
        self.post_lstm_gate_encoder = GatedLinearUnit(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        # self.post_lstm_gate_decoder = GatedLinearUnit(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_add_norm_encoder = AddNorm(self.hparams.hidden_size, trainable_add=False)
        # self.post_lstm_add_norm_decoder = AddNorm(self.hparams.hidden_size, trainable_add=True)
        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # static enrichment and processing past LSTM
        self.static_enrichment = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
        )

        # attention for long-range processing
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=self.hparams.hidden_size, n_head=self.hparams.attention_head_size, dropout=self.hparams.dropout
        )
        self.post_attn_gate_norm = GateAddNorm(
            self.hparams.hidden_size, dropout=self.hparams.dropout, trainable_add=False
        )
        self.pos_wise_ff = GatedResidualNetwork(
            self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.hidden_size, dropout=self.hparams.dropout
        )

        # output processing -> no dropout at this late stage
        self.pre_output_gate_norm = GateAddNorm(self.hparams.hidden_size, dropout=None, trainable_add=False)

        self.output_layer = nn.Linear(self.hparams.hidden_size, self.hparams.output_size)

    @property
    def static_variables(self) -> List[str]:
        return self.hparams.static_categoricals + self.hparams.static_reals

    @property
    def encoder_variables(self) -> List[str]:
        return self.hparams.time_varying_categoricals_encoder + self.hparams.time_varying_reals_encoder

    @property
    def decoder_variables(self) -> List[str]:
        return self.hparams.time_varying_categoricals_decoder + self.hparams.time_varying_reals_decoder

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
            TemporalFusionTransformer
        """
        if allowed_encoder_known_variable_names is None:
            allowed_encoder_known_variable_names = (
                dataset.time_varying_known_categoricals + dataset.time_varying_known_reals
            )

        # embeddings
        embedding_labels = {
            name: encoder.classes_
            for name, encoder in dataset.categorical_encoders.items()
            if name in dataset.categoricals
        }
        embedding_paddings = dataset.dropout_categoricals
        # determine embedding sizes based on heuristic
        embedding_sizes = {
            name: (len(encoder.classes_), get_embedding_size(len(encoder.classes_)))
            for name, encoder in dataset.categorical_encoders.items()
            if name in dataset.categoricals
        }
        embedding_sizes.update(kwargs.get("embedding_sizes", {}))
        kwargs.setdefault("embedding_sizes", embedding_sizes)

        new_kwargs = dict(
            max_encoder_length=dataset.max_encoder_length,
            static_categoricals=dataset.static_categoricals,
            time_varying_categoricals_encoder=[
                name for name in dataset.time_varying_known_categoricals if name in allowed_encoder_known_variable_names
            ]
            + dataset.time_varying_unknown_categoricals,
            time_varying_categoricals_decoder=dataset.time_varying_known_categoricals,
            static_reals=dataset.static_reals,
            time_varying_reals_encoder=[
                name for name in dataset.time_varying_known_reals if name in allowed_encoder_known_variable_names
            ]
            + dataset.time_varying_unknown_reals,
            time_varying_reals_decoder=dataset.time_varying_known_reals,
            x_reals=dataset.reals,
            x_categoricals=dataset.flat_categoricals,
            embedding_labels=embedding_labels,
            embedding_paddings=embedding_paddings,
            categorical_groups=dataset.variable_groups,
        )
        new_kwargs.update(kwargs)

        # create class and return
        return super().from_dataset(dataset, **new_kwargs)

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

    def get_attention_mask(self, encoder_lengths: torch.LongTensor, decoder_length: int):
        """
        Returns causal mask to apply for self-attention layer.

        Args:
            self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        # indices to which is attended
        attend_step = torch.arange(decoder_length, device=self.device)
        # indices for which is predicted
        predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
        # do not attend to steps to self or after prediction
        # todo: there is potential value in attending to future forecasts if they are made with knowledge currently
        #   available
        #   one possibility is here to use a second attention layer for future attention (assuming different effects
        #   matter in the future than the past)
        #   or alternatively using the same layer but allowing forward attention - i.e. only masking out non-available
        #   data and self
        decoder_mask = attend_step > predict_step
        # do not attend to steps where data is padded
        encoder_mask = self._get_mask(encoder_lengths.max(), encoder_lengths)
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(encoder_lengths.size(0), -1, -1),
            ),
            dim=2,
        )
        return mask

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # concatenate in time dimension
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # concatenate in time dimension
        timesteps = x_cont.size(1)  # encode + decode length
        max_encoder_length = int(encoder_lengths.max())
        input_vectors = {}
        for name, emb in self.input_embeddings.items():
            if name in self.hparams.categorical_groups:
                input_vectors[name] = emb(
                    x_cat[
                        ...,
                        [
                            self.hparams.x_categoricals.index(cat_name)
                            for cat_name in self.hparams.categorical_groups[name]
                        ],
                    ]
                )
            else:
                input_vectors[name] = emb(x_cat[..., self.hparams.x_categoricals.index(name)])
        input_vectors.update({name: x_cont[..., idx].unsqueeze(-1) for idx, name in enumerate(self.hparams.x_reals)})

        # Embedding and variable selection
        if len(self.hparams.static_categoricals + self.hparams.static_reals) > 0:
            # static embeddings will be constant over entire batch
            static_embedding = {
                name: input_vectors[name][:, 0] for name in self.hparams.static_categoricals + self.hparams.static_reals
            }
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size), dtype=self.dtype, device=self.device
            )
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length]
            for name in self.hparams.time_varying_categoricals_encoder + self.hparams.time_varying_reals_encoder
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection[:, :max_encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:]  # select decoder
            for name in self.hparams.time_varying_categoricals_decoder + self.hparams.time_varying_reals_decoder
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder,
            static_context_variable_selection[:, max_encoder_length:],
        )

        # LSTM
        # run lstm at least once, i.e. encode length has to be > 0
        lstm_encoder_lengths = encoder_lengths.where(encoder_lengths > 0, torch.ones_like(encoder_lengths))
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1)

        # # run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            rnn.pack_padded_sequence(
                embeddings_varying_encoder, lstm_encoder_lengths, enforce_sorted=False, batch_first=True
            ),
            (input_hidden, input_cell),
        )
        encoder_output, _ = rnn.pad_packed_sequence(encoder_output, batch_first=True)
        # replace hidden cell with initial input if encoder_length is zero to determine correct initial state
        no_encoding = (encoder_lengths == 0)[None, :, None]  # shape: n_lstm_layers x batch_size x hidden_size
        hidden = hidden.masked_scatter(no_encoding, input_hidden)
        cell = cell.masked_scatter(no_encoding, input_cell)

        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            rnn.pack_padded_sequence(
                embeddings_varying_decoder, decoder_lengths, enforce_sorted=False, batch_first=True
            ),
            (hidden, cell),
        )

        decoder_output, _ = rnn.pad_packed_sequence(decoder_output, batch_first=True)

        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
        )

        # skip connection over lstm
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, self.expand_static_context(static_context_enrichment, timesteps)
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths, decoder_length=timesteps - max_encoder_length
            ),
        )

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:])

        output = self.pos_wise_ff(attn_output)

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])
        output = self.output_layer(output)

        return dict(
            prediction=output,
            attention=attn_output_weights,
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
            groups=x["groups"],
            decoder_time_idx=x["decoder_time_idx"],
            target_scale=x["target_scale"],
        )

    def on_train_end(self):
        if self.log_interval(train=True) > 0:
            self._log_embeddings()

    def step(self, x, y, batch_idx, label="train"):
        """
        run at each step for training or validation
        """
        # extract data and run model
        y = rnn.pack_padded_sequence(y, lengths=x["decoder_lengths"], batch_first=True, enforce_sorted=False)
        log, out = super().step(x, y, batch_idx, label=label)
        # calculate interpretations etc for latter logging
        if self.log_interval(label == "train") > 0:
            detached_output = {name: tensor.detach() for name, tensor in out.items()}
            interpretation = self.interpret_output(
                detached_output,
                reduction="sum",
                attention_prediction_horizon=0,  # attention only for first prediction horizon
            )
            log["interpretation"] = interpretation
        return log, out

    def epoch_end(self, outputs, label="train"):
        """
        run at epoch end for training or validation
        """
        log, out = super().epoch_end(outputs, label=label)
        if self.log_interval(label == "train") > 0:
            self._log_interpretation(out, label=label)
        return log, out

    def interpret_output(
        self,
        out: Dict[str, torch.Tensor],
        reduction: str = "none",
        attention_prediction_horizon: int = 0,
        attention_as_autocorrelation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        interpret output of model

        Args:
            out: output as produced by ``forward()``
            reduction: "none" for no averaging over batches, "sum" for summing attentions, "mean" for
                normalizing by encode lengths
            attention_prediction_horizon: which prediction horizon to use for attention
            attention_as_autocorrelation: if to record attention as autocorrelation - this should be set to true in
                case of ``reduction != "none"`` and differing prediction times of the samples. Defaults to False

        Returns:
            interpretations that can be plotted with ``plot_interpretation()``
        """

        # histogram of decode and encode lengths
        encoder_length_histogram = integer_histogram(out["encoder_lengths"], min=0, max=self.hparams.max_encoder_length)
        decoder_length_histogram = integer_histogram(
            out["decoder_lengths"], min=1, max=out["decoder_variables"].size(1)
        )

        # mask where decoder and encoder where not applied when averaging variable selection weights
        encoder_variables = out["encoder_variables"].squeeze(-2)
        encode_mask = self._get_mask(encoder_variables.size(1), out["encoder_lengths"])
        encoder_variables = encoder_variables.masked_fill(encode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        encoder_variables /= (
            out["encoder_lengths"]
            .where(out["encoder_lengths"] > 0, torch.ones_like(out["encoder_lengths"]))
            .unsqueeze(-1)
        )

        decoder_variables = out["decoder_variables"].squeeze(-2)
        decode_mask = self._get_mask(decoder_variables.size(1), out["decoder_lengths"])
        decoder_variables = decoder_variables.masked_fill(decode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        decoder_variables /= out["decoder_lengths"].unsqueeze(-1)

        # static variables need no masking
        static_variables = out["static_variables"].squeeze(1)
        # attention is batch x time x heads x time_to_attend
        # average over heads + only keep prediction attention and attention on observed timesteps
        attention = out["attention"][
            :, attention_prediction_horizon, :, : out["encoder_lengths"].max() + attention_prediction_horizon + 1
        ].mean(1)

        if reduction != "none":  # if to average over batches
            static_variables = static_variables.sum(dim=0)
            encoder_variables = encoder_variables.sum(dim=0)
            decoder_variables = decoder_variables.sum(dim=0)

            # reorder attention or averaging
            for i in range(len(attention)):  # very inefficient but does the trick
                if 0 < out["encoder_lengths"][i] < attention.size(1) - attention_prediction_horizon - 1:
                    relevant_attention = attention[
                        i, : out["encoder_lengths"][i] + attention_prediction_horizon + 1
                    ].clone()
                    if attention_as_autocorrelation:
                        relevant_attention = autocorrelation(relevant_attention)
                    attention[i, -out["encoder_lengths"][i] - attention_prediction_horizon - 1 :] = relevant_attention
                    attention[
                        i, : attention.size(1) - out["encoder_lengths"][i] - attention_prediction_horizon - 1
                    ] = 0.0
                elif attention_as_autocorrelation:
                    attention[i] = autocorrelation(attention[i])

            attention = attention.sum(dim=0)
            if reduction == "mean":
                attention = attention / encoder_length_histogram[1:].flip(0).cumsum(0).clamp(1)
                attention = attention / attention.sum(-1).unsqueeze(-1)  # renormalize
            elif reduction == "sum":
                pass
            else:
                raise ValueError(f"Unknown reduction {reduction}")

            attention = torch.zeros(
                self.hparams.max_encoder_length + attention_prediction_horizon + 1, device=self.device
            ).scatter(
                dim=0,
                index=torch.arange(
                    self.hparams.max_encoder_length + attention_prediction_horizon + 1 - attention.size(-1),
                    self.hparams.max_encoder_length + attention_prediction_horizon + 1,
                    device=self.device,
                ),
                src=attention,
            )
        else:
            attention = attention / attention.sum(-1).unsqueeze(-1)  # renormalize

        interpretation = dict(
            attention=attention,
            static_variables=static_variables,
            encoder_variables=encoder_variables,
            decoder_variables=decoder_variables,
            encoder_length_histogram=encoder_length_histogram,
            decoder_length_histogram=decoder_length_histogram,
        )
        return interpretation

    def plot_prediction(
        self,
        x: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx: int,
        plot_attention: bool = True,
        add_loss_to_title: bool = False,
        ax=None,
    ) -> plt.Figure:
        """
        Plot actuals vs prediction and attention

        Args:
            x (Dict[str, torch.Tensor]): network input
            out (Dict[str, torch.Tensor]): network output
            idx (int): sample index
            plot_attention: if to plot attention on secondary axis
            add_loss_to_title: if to add loss to title. Default to False.
            ax: matplotlib axes to plot on

        Returns:
            plt.Figure: matplotlib figure
        """

        # plot prediction as normal
        fig = super().plot_prediction(x, out, idx=idx, add_loss_to_title=add_loss_to_title, ax=ax)

        # add attention on secondary axis
        if plot_attention:
            interpretation = self.interpret_output(out)
            ax = fig.axes[0]
            ax2 = ax.twinx()
            ax2.set_ylabel("Attention")
            max_encoder_length = out["encoder_lengths"].max()
            encoder_length = x["encoder_lengths"][idx]
            ax2.plot(
                torch.arange(-encoder_length, 1),
                interpretation["attention"][idx, list(range(encoder_length)) + [max_encoder_length]].detach().cpu(),
                alpha=0.2,
                color="k",
            )
        fig.tight_layout()
        return fig

    def plot_interpretation(self, interpretation: Dict[str, torch.Tensor]) -> Dict[str, plt.Figure]:
        """
        Make figures that interpret model.

        * Attention
        * Variable selection weights / importances

        Args:
            interpretation: as obtained from ``interpret_output()``

        Returns:
            dictionary of matplotlib figures
        """
        figs = {}

        # attention
        fig, ax = plt.subplots()
        attention = interpretation["attention"].cpu()
        attention = attention / attention.sum(-1).unsqueeze(-1)
        ax.plot(
            np.arange(-self.hparams.max_encoder_length, attention.size(0) - self.hparams.max_encoder_length), attention
        )
        ax.set_xlabel("Time index")
        ax.set_ylabel("Attention")
        ax.set_title("Attention")
        figs["attention"] = fig

        # variable selection
        def make_selection_plot(title, values, labels):
            fig, ax = plt.subplots(figsize=(7, len(values) * 0.25 + 2))
            order = np.argsort(values)
            values = values / values.sum(-1).unsqueeze(-1)
            ax.barh(np.arange(len(values)), values[order] * 100, tick_label=np.asarray(labels)[order])
            ax.set_title(title)
            ax.set_xlabel("Importance in %")
            plt.tight_layout()
            return fig

        figs["static_variables"] = make_selection_plot(
            "Static variables importance", interpretation["static_variables"].cpu(), self.static_variables
        )
        figs["encoder_variables"] = make_selection_plot(
            "Encoder variables importance", interpretation["encoder_variables"].cpu(), self.encoder_variables
        )
        figs["decoder_variables"] = make_selection_plot(
            "Decoder variables importance", interpretation["decoder_variables"].cpu(), self.decoder_variables
        )

        return figs

    def _log_interpretation(self, outputs, label="train"):
        """
        Log interpretation metrics to tensorboard.
        """
        # extract interpretations
        interpretation = {
            name: torch.stack([x["interpretation"][name] for x in outputs]).sum(0)
            for name in outputs[0]["interpretation"].keys()
        }
        # normalize attention with length histogram squared to account for: 1. zeros in attention and
        # 2. higher attention due to less values
        attention_occurances = interpretation["encoder_length_histogram"][1:].flip(0).cumsum(0).float()
        attention_occurances = attention_occurances / attention_occurances.max()
        attention_occurances = torch.cat(
            [
                attention_occurances,
                torch.ones(
                    interpretation["attention"].size(0) - attention_occurances.size(0),
                    dtype=attention_occurances.dtype,
                    device=attention_occurances.device,
                ),
            ],
            dim=0,
        )
        interpretation["attention"] = interpretation["attention"] / attention_occurances.pow(2).clamp(1.0)
        interpretation["attention"] = interpretation["attention"] / interpretation["attention"].sum()

        figs = self.plot_interpretation(interpretation)  # make interpretation figures
        # log to tensorboard
        for name, fig in figs.items():
            self.logger.experiment.add_figure(
                f"{label.capitalize()} {name} importance", fig, global_step=self.global_step
            )

        # log lengths of encoder/decoder
        for type in ["encoder", "decoder"]:
            fig, ax = plt.subplots()
            lengths = torch.stack([out["interpretation"][f"{type}_length_histogram"] for out in outputs]).sum(0).cpu()
            if type == "decoder":
                start = 1
            else:
                start = 0
            ax.plot(torch.arange(start, start + len(lengths)), lengths)
            ax.set_xlabel(f"{type.capitalize()} length")
            ax.set_ylabel("Number of samples")
            ax.set_title(f"{type.capitalize()} length distribution in {label} epoch")

            self.logger.experiment.add_figure(
                f"{label.capitalize()} {type} length distribution", fig, global_step=self.global_step
            )

    def _log_embeddings(self):
        """
        Log embeddings to tensorboard
        """
        for name, emb in self.input_embeddings.items():
            labels = self.hparams.embedding_labels[name]
            self.logger.experiment.add_embedding(
                emb.weight.data.cpu(), metadata=labels, tag=name, global_step=self.global_step
            )
