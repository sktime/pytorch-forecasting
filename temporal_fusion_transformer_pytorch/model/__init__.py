from typing import List, Union, Dict

import math
import torch
from torch import nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from temporal_fusion_transformer_pytorch.model.sub_modules import (
    GateAddNorm,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
)

from temporal_fusion_transformer_pytorch.model.metrics import QuantileLoss
from temporal_fusion_transformer_pytorch.data import TimeSeriesDataSet


class TemporalFusionTransformer(pl.LightningModule):
    # TODO: support omissions of variables
    # TODO: variable embedding size
    # TODO: refactor
    # TODO: docstrings and comments
    # TODO: asserts
    # TODO: different sequence lengths
    def __init__(
        self,
        encode_length: int = 10,
        target_idx: int = 0,
        hidden_size: int = 8,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        embedding_dim: int = 16,
        output_size: int = 3,
        loss=QuantileLoss([0.1, 0.5, 0.9]),
        attn_heads: int = 4,
        static_categoricals: List[int] = [],
        static_reals: List[int] = [],
        time_varying_categoricals_encoder: List[int] = [],
        time_varying_categoricals_decoder: List[int] = [],
        time_varying_reals_encoder: List[int] = [],
        time_varying_reals_decoder: List[int] = [],
        embedding_size: Dict[str, Union[int, List[int]]] = {},
        learning_rate: float = 1e-3,
        log_interval: int = 25,
        cummulative=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.loss = loss

        # embeddings
        self.input_embeddings = nn.ModuleDict()
        for i in set(
            self.hparams.static_categoricals
            + self.hparams.time_varying_categoricals_encoder
            + self.hparams.time_varying_categoricals_decoder
        ):
            self.input_embeddings[str(i)] = nn.Embedding(
                self.hparams.embedding_size[str(i)], self.hparams.embedding_dim
            )

        # linear layers
        self.input_linear = nn.ModuleDict()
        for i in set(
            self.hparams.time_varying_reals_encoder
            + self.hparams.time_varying_reals_encoder
            + self.hparams.static_reals
        ):
            self.input_linear[str(i)] = nn.Linear(1, self.hparams.embedding_dim)

        # variable selection
        # variable selection for static variables
        self.static_variable_selection = VariableSelectionNetwork(
            input_size=self.hparams.embedding_dim,
            num_inputs=len(self.hparams.static_categoricals) + len(self.hparams.static_reals),
            hidden_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # variable selection for encoder
        self.encoder_variable_selection = VariableSelectionNetwork(
            input_size=self.hparams.embedding_dim,
            num_inputs=len(self.hparams.time_varying_reals_encoder)
            + len(self.hparams.time_varying_categoricals_encoder),
            hidden_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context=self.hparams.hidden_size,
        )

        # variable selection for decoder
        self.decoder_variable_selection = VariableSelectionNetwork(
            input_size=self.hparams.embedding_dim,
            num_inputs=len(self.hparams.time_varying_reals_decoder)
            + len(self.hparams.time_varying_categoricals_decoder),
            hidden_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context=self.hparams.hidden_size,
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
            context=self.hparams.hidden_size,
        )

        # lstm encoder (history) and decoder (future)
        self.lstm_encoder = nn.LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout,
        )

        self.lstm_decoder = nn.LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout,
        )

        # skip connection for lstm
        self.post_lstm_gate_norm = GateAddNorm(self.hparams.hidden_size, dropout=self.hparams.dropout)

        # attention
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=self.hparams.hidden_size, n_head=self.hparams.attn_heads, dropout=self.hparams.dropout
        )
        self.post_attn_gate_norm = GateAddNorm(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.pos_wise_ff = GatedResidualNetwork(
            self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.hidden_size, self.hparams.dropout
        )

        # output processing
        self.pre_output_gate_norm = GateAddNorm(self.hparams.hidden_size, dropout=self.hparams.dropout)

        self.output_layer = nn.Linear(self.hparams.hidden_size, self.hparams.output_size)

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):

        # categoricals
        length = len(dataset.static_categoricals)
        start = 0
        static_categoricals = list(range(start, length))
        length = len(dataset.time_varying_known_categoricals)
        time_varying_known_categoricals = list(range(start, start + length))
        start += length
        length = len(dataset.time_varying_unknown_categoricals)
        time_varying_unknown_categoricals = list(range(start, start + length))
        kwargs.setdefault(
            "embedding_size",
            {
                str(idx): len(dataset.categoricals_encoders[name].classes_)
                for idx, name in enumerate(dataset.categoricals)
            },
        )
        # reals
        length = len(dataset.static_reals)
        start = 0
        static_reals = list(range(start, length))
        length = len(dataset.time_varying_known_reals)
        time_varying_known_reals = list(range(start, start + length))
        start += length
        length = len(dataset.time_varying_unknown_reals)
        time_varying_unknown_reals = list(range(start, start + length))
        target_idx = start + dataset.time_varying_unknown_reals.index(dataset.target)

        return cls(
            encode_length=dataset.max_encode_length,
            static_categoricals=static_categoricals,
            time_varying_categoricals_encoder=time_varying_known_categoricals + time_varying_unknown_categoricals,
            time_varying_categoricals_decoder=time_varying_known_categoricals,
            static_reals=static_reals,
            time_varying_reals_encoder=time_varying_known_reals + time_varying_unknown_reals,
            time_varying_reals_decoder=time_varying_known_reals,
            target_idx=target_idx,
            **kwargs,
        )

    def expand_static_context(self, context):
        return context[:, None].expand(-1, self._timesteps, -1).transpose(0, 1)

    def get_decoder_mask(self, self_attn_inputs):
        """Returns causal mask to apply for self-attention layer.
        Args:
        self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        len_s = self_attn_inputs.shape[1]
        bs = self_attn_inputs.shape[0]
        mask = torch.cumsum(torch.eye(len_s), 0)
        mask = mask.repeat(bs, 1, 1).float()
        return mask

    def forward(self, x_cat, x_cont=None):
        """
        input dimensions: n_samples x time x variables

        ##inputs should be in this order
        # static categorical
        # time_varying_categorical_only_past
        # time_varying_categorical_past_and_future

        # static_real
        # time_varying_real_only_past
        # time_varying_real_past_and_future
        """
        self._timesteps = x_cat.size(1)
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
        static_embedding, static_sparse_weights = self.static_variable_selection(static_embedding)

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding)
        )

        embeddings_varying_encoder = torch.cat(
            [embedding_vectors[i] for i in self.hparams.time_varying_categoricals_encoder]
            + [continuous_vectors[i] for i in self.hparams.time_varying_reals_encoder],
            dim=2,
        ).transpose(0, 1)[: self.hparams.encode_length]
        embeddings_varying_decoder = torch.cat(
            [embedding_vectors[i] for i in self.hparams.time_varying_categoricals_decoder]
            + [continuous_vectors[i] for i in self.hparams.time_varying_reals_decoder],
            dim=2,
        ).transpose(0, 1)[self.hparams.encode_length :]
        (embeddings_varying_encoder, encoder_sparse_weights,) = self.encoder_variable_selection(
            embeddings_varying_encoder, static_context_variable_selection[: self.hparams.encode_length],
        )

        (embeddings_varying_decoder, decoder_sparse_weights,) = self.decoder_variable_selection(
            embeddings_varying_decoder, static_context_variable_selection[self.hparams.encode_length :],
        )
        # LSTM
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder,
            (
                self.static_context_initial_hidden_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1),
                self.static_context_initial_cell_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1),
            ),
        )
        decoder_output, _ = self.lstm_decoder(embeddings_varying_decoder, (hidden, cell))
        lstm_output = torch.cat([encoder_output, decoder_output], dim=0)

        # skip connection over lstm
        lstm_output = self.post_lstm_gate_norm(
            lstm_output, torch.cat([embeddings_varying_encoder, embeddings_varying_decoder], dim=0)
        )

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(lstm_output, self.expand_static_context(static_context_enrichment))

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            attn_input, attn_input, attn_input, mask=self.get_decoder_mask(attn_input)
        )

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input)

        output = self.pos_wise_ff(attn_output[self.hparams.encode_length :])

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[self.hparams.encode_length :])
        output = self.output_layer(output.transpose(0, 1))

        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(**x)
        y_all = x["x_cont"][..., self.hparams.target_idx]

        # prediction is supposed to be cummulative
        loss = self.loss(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        # log prediction figure
        if batch_idx % self.hparams.log_interval == 0:
            fig = self.plot_prediction(y_all[0], y_hat[0].detach().cpu())  # first in batch
            self.logger.experiment.add_figure(
                "Training prediction", fig, global_step=self.global_step,
            )
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        # log loss
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_loss}
        return {"train_loss": avg_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(**x)
        y_all = x["x_cont"][..., self.hparams.target_idx]

        loss = self.loss(y_hat, y, cummulative=self.hparams.cummulative)

        log = {"val_loss": loss}
        # log prediction figure
        if batch_idx % self.hparams.log_interval == 0:
            fig = self.plot_prediction(y_all[0], y_hat[0].detach().cpu())  # first in batch
            self.logger.experiment.add_figure(
                f"Validation prediction of item 0 in batch {batch_idx}", fig, global_step=self.global_step,
            )
        return log

    def validation_epoch_end(self, outputs):
        # loss logging
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def plot_prediction(self, y, y_hat):
        fig, ax = plt.subplots()
        y_pred = y_hat
        n_pred = y_pred.shape[0]
        x_obs = np.arange(y.shape[0] - n_pred)
        x_pred = np.arange(y.shape[0] - n_pred, y.shape[0])
        prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
        obs_color = next(prop_cycle)["color"]
        ax.plot(x_obs, y[:-n_pred], label="observed", c=obs_color)
        ax.plot(x_pred, y[-n_pred:], label=None, c=obs_color)
        for i in range(y_pred.shape[-1]):
            ax.plot(x_pred, y_pred[:, i], label=f"predicted {i}", c=next(prop_cycle)["color"])
        loss = self.loss(y.unsqueeze(0), y_pred.unsqueeze(0), cummulative=self.hparams.cummulative)
        ax.set_title(f"Loss {loss:.3g}")
        fig.legend()
        return fig

    def plot_attention(self):
        pass

    def plot_importance(self):
        pass

    def on_after_backward(self):
        if self.global_step % self.hparams.log_interval == 0:
            self._log_grad_flow(self.named_parameters())

    def _log_grad_flow(self, named_parameters):
        """
        log distribution of gradients to identify exploding / vanishing gradients
        """
        ave_grads = []
        layers = []
        for name, p in named_parameters:
            if p.grad is not None:
                if p.requires_grad and "bias" not in name:
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
