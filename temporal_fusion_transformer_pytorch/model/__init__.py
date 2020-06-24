from typing import List, Union, Dict, Tuple

import math
import torch
from torch import nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils import rnn

from temporal_fusion_transformer_pytorch.model.sub_modules import (
    GateAddNorm,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
)

from temporal_fusion_transformer_pytorch.model.metrics import QuantileLoss
from temporal_fusion_transformer_pytorch.data import TimeSeriesDataSet
from temporal_fusion_transformer_pytorch.utils import integer_histogram
from pytorch_ranger import Ranger


class TemporalFusionTransformer(pl.LightningModule):
    # TODO: dependence plot logging
    # todo: weights
    # todo: poisson/negative binomial
    # todo: prediction with df
    def __init__(
        self,
        hidden_size: int = 16,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        output_size: int = 5,
        loss=QuantileLoss([0.1, 0.25, 0.5, 0.75, 0.9]),
        attention_head_size: int = 4,
        max_encode_length: int = 10,
        target_idx: int = 0,
        static_categoricals: List[int] = [],
        static_reals: List[int] = [],
        time_varying_categoricals_encoder: List[int] = [],
        time_varying_categoricals_decoder: List[int] = [],
        time_varying_reals_encoder: List[int] = [],
        time_varying_reals_decoder: List[int] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_labels: Dict[str, np.ndarray] = {},
        real_labels: Dict[str, str] = {},
        categorical_labels: Dict[str, str] = {},
        learning_rate: float = 1e-3,
        log_interval: int = 10,
        log_gradient_flow: bool = False,
    ):
        """
        Temporal Fusion Transformer for forecasting timeseries. Use ``from_dataset()`` to 

        Args:

            hidden_size: hidden size of network which is its main hyperparameter and can range from 8 to 512
            lstm_layers: number of LSTM layers (2 is mostly optimal)
            dropout: dropout rate
            hidden_continuous_size: hidden size for processing continous variables (similar to categorical
                embedding size)
            output_size: output size (e.g. for multiple quantiles as outputs)
            loss: loss function taking prediction and targets
            attention_head_size: number of attention heads (4 is a good default)
            max_encode_length: length to encode
            target_idx: index of continuous target variable
            static_categoricals: integer of positions of static categorical variables
            static_reals: integer of positions of static continuous variables
            time_varying_categoricals_encoder: integer of positions of categorical variables for encoder
            time_varying_categoricals_decoder: integer of positions of categorical variables for decoder
            time_varying_reals_encoder: integer of positions of continuous variables for encoder
            time_varying_reals_decoder: integer of positions of continuous variables for decoder
            embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
                embedding size
            embedding_labels: dictionary mapping (string) indices to list of categorical labels
            real_labels: dictionary mapping (string) indices to continuous variable names
            categorical_labels: dictionary mapping (string) indices to categorical variable names
            learning_rate: learning rate
            log_interval: log predictions every x batches, do not log if 0 or less, log interpretation if > 0
            log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
                failures
        """
        super().__init__()
        self.save_hyperparameters()  # store all arguments
        # store loss function separately as it is a module
        self.loss = loss

        # proessing inputs
        # embeddings
        self.input_embeddings = nn.ModuleDict()
        for i in set(
            self.hparams.static_categoricals
            + self.hparams.time_varying_categoricals_encoder
            + self.hparams.time_varying_categoricals_decoder
        ):
            self.input_embeddings[str(i)] = nn.Embedding(*self.hparams.embedding_sizes[str(i)])

        # linear layers
        self.input_linear = nn.ModuleDict()
        for i in set(
            self.hparams.time_varying_reals_encoder
            + self.hparams.time_varying_reals_encoder
            + self.hparams.static_reals
        ):
            self.input_linear[str(i)] = nn.Linear(1, self.hparams.hidden_continuous_size)

        # variable selection
        # variable selection for static variables
        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=[self.hparams.embedding_sizes[str(i)][1] for i in self.hparams.static_categoricals]
            + [self.hparams.hidden_continuous_size for _ in self.hparams.static_reals],
            hidden_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # variable selection for encoder
        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=[
                self.hparams.embedding_sizes[str(i)][1] for i in self.hparams.time_varying_categoricals_encoder
            ]
            + [self.hparams.hidden_continuous_size for _ in self.hparams.time_varying_reals_encoder],
            hidden_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
        )

        # variable selection for decoder
        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=[
                self.hparams.embedding_sizes[str(i)][1] for i in self.hparams.time_varying_categoricals_decoder
            ]
            + [self.hparams.hidden_continuous_size for _ in self.hparams.time_varying_reals_decoder],
            hidden_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
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
        self.static_enrichment = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
        )

        # lstm encoder (history) and decoder (future) for local processing
        self.lstm_encoder = nn.LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout,
            batch_first=True,
        )

        self.lstm_decoder = nn.LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout,
            batch_first=True,
        )

        # skip connection for lstm
        self.post_lstm_gate_norm = GateAddNorm(self.hparams.hidden_size, dropout=self.hparams.dropout)

        # attention for long-range processing
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=self.hparams.hidden_size, n_head=self.hparams.attention_head_size, dropout=self.hparams.dropout
        )
        self.post_attn_gate_norm = GateAddNorm(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.pos_wise_ff = GatedResidualNetwork(
            self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.dropout
        )

        # output processing
        self.pre_output_gate_norm = GateAddNorm(self.hparams.hidden_size, dropout=self.hparams.dropout)

        self.output_layer = nn.Linear(self.hparams.hidden_size, self.hparams.output_size)

    @property
    def static_variables(self) -> List[str]:
        return [self.hparams.categorical_labels[str(i)] for i in self.hparams.static_categoricals] + [
            self.hparams.real_labels[str(i)] for i in self.hparams.static_reals
        ]

    @property
    def encoder_variables(self) -> List[str]:
        return [self.hparams.categorical_labels[str(i)] for i in self.hparams.time_varying_categoricals_encoder] + [
            self.hparams.real_labels[str(i)] for i in self.hparams.time_varying_reals_encoder
        ]

    @property
    def decoder_variables(self) -> List[str]:
        return [self.hparams.categorical_labels[str(i)] for i in self.hparams.time_varying_categoricals_decoder] + [
            self.hparams.real_labels[str(i)] for i in self.hparams.time_varying_reals_decoder
        ]

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        """
        create model from dataset

        Args:
            dataset:
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            TemporalFusionTransformer
        """
        start = 0
        length = len(dataset.static_categoricals)
        static_categoricals = list(range(start, length))

        start += length
        length = len(dataset.time_varying_known_categoricals)
        time_varying_known_categoricals = list(range(start, start + length))

        start += length
        length = len(dataset.time_varying_unknown_categoricals)
        time_varying_unknown_categoricals = list(range(start, start + length))

        categorical_labels = {
            str(idx): name
            for idx, name in enumerate(
                dataset.static_categoricals
                + dataset.time_varying_known_categoricals
                + dataset.time_varying_unknown_categoricals
            )
        }
        embedding_labels = {
            str(idx): dataset.categoricals_encoders[name].classes_ for idx, name in enumerate(dataset.categoricals)
        }
        # determine embedding sizes based on heuristic
        kwargs.setdefault(
            "embedding_sizes",
            {idx: (len(labels), round(1.6 * len(labels) ** 0.56)) for idx, labels in embedding_labels.items()},
        )
        # reals
        start = 0
        length = len(dataset.static_reals)
        static_reals = list(range(start, length))

        start += length
        length = len(dataset.time_varying_known_reals)
        time_varying_known_reals = list(range(start, start + length))

        start += length
        length = len(dataset.time_varying_unknown_reals)
        time_varying_unknown_reals = list(range(start, start + length))

        real_labels = {
            str(idx): name
            for idx, name in enumerate(
                dataset.static_reals + dataset.time_varying_known_reals + dataset.time_varying_unknown_reals
            )
        }
        target_idx = len(dataset.reals)

        # create class and return
        return cls(
            max_encode_length=dataset.max_encode_length,
            static_categoricals=static_categoricals,
            time_varying_categoricals_encoder=time_varying_known_categoricals + time_varying_unknown_categoricals,
            time_varying_categoricals_decoder=time_varying_known_categoricals,
            static_reals=static_reals,
            time_varying_reals_encoder=time_varying_known_reals + time_varying_unknown_reals,
            time_varying_reals_decoder=time_varying_known_reals,
            target_idx=target_idx,
            real_labels=real_labels,
            categorical_labels=categorical_labels,
            embedding_labels=embedding_labels,
            **kwargs,
        )

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

    def get_attention_mask(self, encode_lengths: torch.LongTensor, decode_length: int):
        """Returns causal mask to apply for self-attention layer.
        Args:
        self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        # indices to which is attended
        attend_step = torch.arange(decode_length, device=self.device)
        # indices for which is predicted
        predict_step = torch.arange(decode_length, 0, step=-1, device=self.device)[:, None]
        # do not attend to steps after to prediction
        decoder_mask = attend_step >= predict_step
        # do not attend to steps where data is padded
        encoder_mask = torch.arange(encode_lengths.max(), device=self.device)[None, :] >= encode_lengths[:, None]
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decode_length, -1),
                decoder_mask.unsqueeze(0).expand(encode_lengths.size(0), -1, -1),
            ),
            dim=2,
        )

        return mask

    def forward(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.LongTensor,
        encode_lengths: torch.LongTensor,
        decode_lengths: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        timesteps = x_cat.size(1)  # encode + decode length
        max_encode_length = int(encode_lengths.max())
        embedding_vectors = {int(i): emb(x_cat[..., int(i)]) for i, emb in self.input_embeddings.items()}
        continuous_vectors = {
            int(i): lin(x_cont[..., int(i)].view(x_cont.size(0), -1, 1)) for i, lin in self.input_linear.items()
        }

        # Embedding and variable selection
        static_embedding = torch.cat(
            [embedding_vectors[i][:, 0] for i in self.hparams.static_categoricals]
            + [continuous_vectors[i][:, 0] for i in self.hparams.static_reals],
            dim=1,
        )
        static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        embeddings_varying_encoder = torch.cat(
            [embedding_vectors[i] for i in self.hparams.time_varying_categoricals_encoder]
            + [continuous_vectors[i] for i in self.hparams.time_varying_reals_encoder],
            dim=2,
        )[:, :max_encode_length]
        embeddings_varying_decoder = torch.cat(
            [embedding_vectors[i] for i in self.hparams.time_varying_categoricals_decoder]
            + [continuous_vectors[i] for i in self.hparams.time_varying_reals_decoder],
            dim=2,
        )[:, max_encode_length:]

        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder, static_context_variable_selection[:, :max_encode_length],
        )

        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder, static_context_variable_selection[:, max_encode_length:],
        )
        # LSTM
        # run lstm at least once, i.e. encode length has to be > 0
        lstm_encode_lengths = encode_lengths.where(encode_lengths > 0, torch.ones_like(encode_lengths))
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1)

        # run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            rnn.pack_padded_sequence(
                embeddings_varying_encoder, lstm_encode_lengths, enforce_sorted=False, batch_first=True
            ),
            (input_hidden, input_cell),
        )
        encoder_output, _ = rnn.pad_packed_sequence(encoder_output, batch_first=True)
        # replace hidden cell with initial input if encode_length is zero to determine correct initial state
        no_encoding = (encode_lengths > 0)[None, :, None]
        hidden = hidden.masked_scatter(no_encoding, input_hidden)
        cell = cell.masked_scatter(no_encoding, input_cell)

        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            rnn.pack_padded_sequence(
                embeddings_varying_decoder, decode_lengths, enforce_sorted=False, batch_first=True
            ),
            (hidden, cell),
        )

        decoder_output, _ = rnn.pad_packed_sequence(decoder_output, batch_first=True)
        lstm_output = torch.cat([encoder_output, decoder_output], dim=1)

        # skip connection over lstm
        lstm_output = self.post_lstm_gate_norm(
            lstm_output, torch.cat([embeddings_varying_encoder, embeddings_varying_decoder], dim=1)
        )

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, self.expand_static_context(static_context_enrichment, timesteps)
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encode_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(encode_lengths=encode_lengths, decode_length=timesteps - max_encode_length,),
        )

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encode_length:])

        output = self.pos_wise_ff(attn_output)

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encode_length:])
        output = self.output_layer(output)

        return dict(
            prediction=output,
            attention=attn_output_weights,
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decode_lengths=decode_lengths,
            encode_lengths=encode_lengths,
        )

    def configure_optimizers(self):
        return Ranger(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, label="train")

    def on_after_backward(self):
        if (
            self.global_step % self.hparams.log_interval == 0
            and self.hparams.log_interval > 0
            and self.hparams.log_gradient_flow
        ):
            self._log_gradient_flow(self.named_parameters())

    def training_epoch_end(self, outputs):
        return self._epoch_end(outputs, label="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, label="val")

    def validation_epoch_end(self, outputs):
        return self._epoch_end(outputs, label="val")

    def on_train_end(self):
        if self.hparams.log_interval > 0:
            self._log_embeddings()

    def _step(self, batch, batch_idx, label="train"):
        """
        run at each step for training or validation
        """
        # extract data and run model
        x, y = batch
        out = self(**x)
        y_hat = out["prediction"]
        # calculate loss and log it
        loss = self.loss(y_hat, y)
        tensorboard_logs = {f"{label}_loss": loss}

        if label == "train":
            loss_label = "loss"
        else:
            loss_label = f"{label}_loss"
        log = {
            loss_label: loss,
            "log": tensorboard_logs,
        }

        # calculate interpretations etc for latter logging
        if self.hparams.log_interval > 0:
            detached_output = {name: tensor.detach().cpu() for name, tensor in out.items()}
            interpretation = self.interpret_output(
                detached_output,
                average_batches=True,
                attention_prediction_horizon=0,  # attention only for first prediction horizon
            )
            log["interpretation"] = interpretation
            # histogram of decode and encode lengths
            log["encode_length_histogram"] = integer_histogram(
                x["encode_lengths"], min=0, max=self.hparams.max_encode_length
            )
            log["decode_length_histogram"] = integer_histogram(x["decode_lengths"], min=1, max=y_hat.size(1))

        # log single prediction figure
        if batch_idx % self.hparams.log_interval == 0 and self.hparams.log_interval > 0:
            y_all = x["x_cont"][0, ..., self.hparams.target_idx]  # all true values for y of the first sample in batch
            max_encode_length = x["encode_lengths"].max()
            fig = self.plot_prediction(
                torch.cat(
                    (
                        y_all[: x["encode_lengths"][0]],
                        y_all[max_encode_length : (max_encode_length + x["decode_lengths"][0])],
                    ),
                ),
                y_hat[0, : x["decode_lengths"][0]].detach().cpu(),
            )  # first in batch
            tag = f"{label.capitalize()} prediction"
            if label == "train":
                tag += f" of item 0 in global batch {self.global_step}"
            else:
                tag += f" of item 0 in batch {batch_idx}"
            self.logger.experiment.add_figure(
                tag, fig, global_step=self.global_step,
            )
        return log

    def _epoch_end(self, outputs, label="train"):
        """
        run at epoch end for training or validation
        """
        # log loss
        avg_loss = torch.stack([x[f"{label}_loss"] for x in outputs]).mean()
        if self.hparams.log_interval > 0:
            self._log_interpretation(outputs, label=label)
            self._log_encode_decode_lengths(outputs, label=label)
        tensorboard_logs = {f"avg_{label}_loss": avg_loss}
        return {f"{label}_loss": avg_loss, "log": tensorboard_logs}

    def interpret_output(
        self, out: Dict[str, torch.Tensor], average_batches: bool = False, attention_prediction_horizon: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        interpret output of model

        Args:
            out: output as produced by ``forward()``
            average_batches: if to average over all batches
            attention_prediction_horizon: which prediction horizon to use for attention

        Returns:
            interpretations that can be plotted with ``plot_interpretation()``
        """
        # mask where decoder and encoder where not applied when averaging variable selection weights
        encoder_variables = out["encoder_variables"].squeeze()
        encode_mask = torch.arange(encoder_variables.size(1)).unsqueeze(0) >= out["encode_lengths"].unsqueeze(-1)
        encoder_variables = encoder_variables.masked_fill(encode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        encoder_variables /= (
            out["encode_lengths"].where(out["encode_lengths"] > 0, torch.ones_like(out["encode_lengths"])).unsqueeze(-1)
        )

        decoder_variables = out["decoder_variables"].squeeze()
        decode_mask = torch.arange(decoder_variables.size(1)).unsqueeze(0) >= out["decode_lengths"].unsqueeze(-1)
        decoder_variables = decoder_variables.masked_fill(decode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        decoder_variables /= out["decode_lengths"].unsqueeze(-1)

        # static variables need no masking
        static_variables = out["static_variables"].squeeze()
        # attention is batch x time x heads x time_to_attend
        # average over heads + only keep prediction attention and attention on observed timesteps
        attention = out["attention"][:, attention_prediction_horizon, :, : out["encode_lengths"].max()].mean(1)
        # reoder attention
        for i in range(len(attention)):  # very inefficient but does the trick
            if 0 < out["encode_lengths"][i] < attention.size(1):
                attention[i, -out["encode_lengths"][i] :] = attention[i, : out["encode_lengths"][i]].clone()
                attention[i, : attention.size(1) - out["encode_lengths"][i]] = 0.0

        if average_batches:  # if to average over batches
            static_variables = static_variables.mean(dim=0)
            encoder_variables = encoder_variables.mean(dim=0)
            decoder_variables = decoder_variables.mean(dim=0)
            attention = attention.mean(dim=0)
            attention = attention / attention.sum(-1).unsqueeze(-1)  # renormalize

            attention = torch.zeros(self.hparams.max_encode_length).scatter(
                dim=0,
                index=torch.arange(self.hparams.max_encode_length - attention.size(0), self.hparams.max_encode_length),
                src=attention,
            )
        else:
            attention = attention / attention.sum(-1).unsqueeze(-1)  # renormalize
            attention = torch.zeros(attention.size(0), self.hparams.max_encode_length).scatter(
                dim=1,
                index=torch.arange(
                    self.hparams.max_encode_length - attention.size(0), self.hparams.max_encode_length
                ).unsqueeze(0),
                src=attention,
            )

        interpretation = dict(
            attention=attention,
            static_variables=static_variables,
            encoder_variables=encoder_variables,
            decoder_variables=decoder_variables,
        )
        return interpretation

    def plot_interpretation(self, interpretation: Dict[str, torch.Tensor]) -> Dict[str, plt.Figure]:
        """
        make figures that interpret model:

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
        ax.plot(np.arange(-self.hparams.max_encode_length, 0), interpretation["attention"])
        ax.set_xlabel("Time index")
        ax.set_ylabel("Attention")
        ax.set_title("Attention")
        figs["attention"] = fig

        # variable selection
        def make_selection_plot(title, values, labels):
            fig, ax = plt.subplots(figsize=(7, len(values) * 0.25 + 2))
            order = np.argsort(values)
            ax.barh(np.arange(len(values)), values[order] * 100, tick_label=np.asarray(labels)[order])
            ax.set_title(title)
            ax.set_xlabel("Importance in %")
            plt.tight_layout()
            return fig

        figs["static_variables"] = make_selection_plot(
            "Static variables importance", interpretation["static_variables"], self.static_variables
        )
        figs["encoder_variables"] = make_selection_plot(
            "Encoder variables importance", interpretation["encoder_variables"], self.encoder_variables
        )
        figs["decoder_variables"] = make_selection_plot(
            "Decoder variables importance", interpretation["decoder_variables"], self.decoder_variables
        )

        return figs

    def _log_interpretation(self, outputs, label="train"):
        """
        log interpretation metrics to tensorboard
        """
        # extract interpretations
        interpretation = {
            name: torch.stack([x["interpretation"][name] for x in outputs]).mean(0)
            for name in outputs[0]["interpretation"].keys()
        }
        figs = self.plot_interpretation(interpretation)  # make interpretation figures
        # log to tensorboard
        for name, fig in figs.items():
            self.logger.experiment.add_figure(
                f"{label.capitalize()} {name} importance", fig, global_step=self.global_step
            )

    def _log_encode_decode_lengths(self, outputs, label="train"):
        """
        log decode and encode lengths in histograms on tensorboard
        """
        for type in ["encode", "decode"]:
            fig, ax = plt.subplots()
            lengths = torch.stack([out[f"{type}_length_histogram"] for out in outputs]).sum(0)
            if type == "decode":
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

    def plot_prediction(self, y: torch.Tensor, y_hat: torch.Tensor) -> plt.Figure:
        """
        plot prediction of prediction vs actuals

        Args:
            y: all actual values
            y_hat: predictions

        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots()
        n_pred = y_hat.shape[0]
        x_obs = np.arange(y.shape[0] - n_pred)
        x_pred = np.arange(y.shape[0] - n_pred, y.shape[0])
        prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
        obs_color = next(prop_cycle)["color"]
        pred_color = next(prop_cycle)["color"]
        if len(x_obs) > 0:
            if len(x_obs) > 1:
                plotter = ax.plot
            else:
                plotter = ax.scatter
            plotter(x_obs, y[:-n_pred], label="observed", c=obs_color)
        if len(x_pred) > 1:
            plotter = ax.plot
        else:
            plotter = ax.scatter
        plotter(x_pred, y[-n_pred:], label=None, c=obs_color)
        if isinstance(self.loss, QuantileLoss):
            plotter(x_pred, y_hat[:, y_hat.size(1) // 2], label=f"predicted", c=pred_color)
            for i in range(y_hat.size(1) // 2):
                if len(x_pred) > 1:
                    ax.fill_between(x_pred, y_hat[:, i], y_hat[:, -i - 1], alpha=0.15, fc=pred_color)
                else:
                    ax.errorbar(
                        x_pred, torch.tensor([[y_hat[0, i]], [y_hat[0, -i - 1]]]), c=pred_color, capsize=1.0,
                    )
        else:
            for i in range(y_hat.shape[-1]):
                plotter(x_pred, y_hat[:, i], label=f"predicted {i}", c=next(prop_cycle)["color"])
        loss = self.loss(y_hat.unsqueeze(0), y[-n_pred:].unsqueeze(0))
        ax.set_title(f"Loss {loss:.3g}")
        ax.set_xlabel("Time index")
        fig.legend()
        return fig

    def _log_gradient_flow(self, named_parameters):
        """
        log distribution of gradients to identify exploding / vanishing gradients
        """
        ave_grads = []
        layers = []
        for name, p in named_parameters:
            if p.grad is not None and p.requires_grad and "bias" not in name:
                layers.append(name)
                ave_grads.append(p.grad.abs().mean())
                self.logger.experiment.add_histogram(tag=name, values=p.grad, global_step=self.global_step)
        fig, ax = plt.subplots()
        ax.plot(ave_grads)
        ax.set_xlabel("Layers")
        ax.set_ylabel("Average gradient")
        ax.set_yscale("log")
        ax.set_title("Gradient flow")
        self.logger.experiment.add_figure(f"Gradient flow", fig, global_step=self.global_step)

    def _log_embeddings(self):
        """
        log embeddings to tensorboard
        """
        for idx, emb in self.input_embeddings.items():
            name = self.hparams.categorical_labels[idx]
            labels = self.hparams.embedding_labels[idx]
            data = emb.weight.data
            self.logger.experiment.add_embedding(data, metadata=labels, tag=name, global_step=self.global_step)

    def size(self) -> int:
        """
        get number of parameters in model
        """
        return sum(p.numel() for p in self.parameters())
