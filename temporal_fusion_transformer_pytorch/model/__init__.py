from typing import List, Union, Dict
import torch
from torch import nn
import pytorch_lightning as pl

from temporal_fusion_transformer_pytorch.model.sub_modules import (
    GateAddNorm,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
)


class TemporalFusionTransformer(pl.LightningModule):
    # TODO: support omissions of variables
    # TODO: variable embedding size
    # TODO: refactor
    # TODO: docstrings and comments
    # TODO: asserts
    # TODO: different sequence lengths
    # TODO: add tensorboard logging -> metrics, examples x 20, partial dependence plots
    def __init__(
        self,
        encode_length: int,
        lstm_hidden_dimension: int = 8,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        embedding_dim: int = 7,
        num_quantiles: int = 3,
        attn_heads: int = 2,
        seq_length: int = 100,
        static_categorical_variables: List[int] = [],
        static_real_variables: List[int] = [],
        time_varying_categorical_variables_encoder: List[int] = [],
        time_varying_categorical_variables_decoder: List[int] = [],
        time_varying_real_variables_encoder: List[int] = [],
        time_varying_real_variables_decoder: List[int] = [],
        embedding_size: Dict[str, Union[int, List[int]]] = {},
    ):
        super().__init__()
        self.static_categorical_variables = static_categorical_variables
        self.static_real_variables = static_real_variables
        self.encode_length = encode_length
        self.time_varying_categorical_variables_encoder = time_varying_categorical_variables_encoder
        self.time_varying_categorical_variables_decoder = time_varying_categorical_variables_decoder
        self.time_varying_real_variables_encoder = time_varying_real_variables_encoder
        self.time_varying_real_variables_decoder = time_varying_real_variables_decoder
        self.embedding_size = embedding_size
        self.hidden_size = lstm_hidden_dimension
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.attn_heads = attn_heads
        self.num_quantiles = num_quantiles
        self.seq_length = seq_length

        # consolidate variables

        # embeddings
        self.input_embeddings = nn.ModuleDict()
        for i in set(
            self.static_categorical_variables
            + self.time_varying_categorical_variables_encoder
            + self.time_varying_categorical_variables_decoder
        ):
            self.input_embeddings[str(i)] = nn.Embedding(embedding_size[i], self.embedding_dim)

        # linear layers
        self.input_linear = nn.ModuleDict()
        for i in set(
            self.time_varying_real_variables_encoder
            + self.time_varying_real_variables_encoder
            + self.static_real_variables
        ):
            self.input_linear[str(i)] = nn.Linear(1, self.embedding_dim)

        # variable selection
        # variable selection for static variables
        self.static_variable_selection = VariableSelectionNetwork(
            input_size=self.embedding_dim,
            num_inputs=len(self.static_categorical_variables) + len(self.static_real_variables),
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        )

        # variable selection for encoder
        self.encoder_variable_selection = VariableSelectionNetwork(
            input_size=self.embedding_dim,
            num_inputs=len(self.time_varying_real_variables_encoder)
            + len(self.time_varying_categorical_variables_encoder),
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            context=self.hidden_size,
        )

        # variable selection for decoder
        self.decoder_variable_selection = VariableSelectionNetwork(
            input_size=self.embedding_dim,
            num_inputs=len(self.time_varying_real_variables_decoder)
            + len(self.time_varying_categorical_variables_decoder),
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            context=self.hidden_size,
        )

        # static encoders
        # for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )

        # for hidden state of the lstm
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )

        # for cell state of the lstm
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            self.hidden_size, self.hidden_size, self.hidden_size, self.dropout
        )
        self.static_enrichment = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            context=self.hidden_size,
        )

        # lstm encoder (history) and decoder (future)
        self.lstm_encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout,
        )

        self.lstm_decoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout,
        )

        # skip connection for lstm
        self.post_lstm_gate_norm = GateAddNorm(self.hidden_size, dropout=self.dropout)

        # attention
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=self.hidden_size, n_head=self.attn_heads, dropout=dropout
        )
        self.post_attn_gate_norm = GateAddNorm(self.hidden_size, dropout=self.dropout)
        self.pos_wise_ff = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout)

        # output processing
        self.pre_output_gate_norm = GateAddNorm(self.hidden_size, dropout=self.dropout)

        self.output_layer = nn.Linear(self.hidden_size, self.num_quantiles)

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
            [embedding_vectors[i][:, 0] for i in self.static_categorical_variables]
            + [continuous_vectors[i][:, 0] for i in self.static_real_variables],
            dim=1,
        )
        static_embedding, static_sparse_weights = self.static_variable_selection(static_embedding)

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding)
        )

        embeddings_varying_encoder = torch.cat(
            [embedding_vectors[i] for i in self.time_varying_categorical_variables_encoder]
            + [continuous_vectors[i] for i in self.time_varying_real_variables_encoder],
            dim=2,
        ).transpose(0, 1)[: self.encode_length]
        embeddings_varying_decoder = torch.cat(
            [embedding_vectors[i] for i in self.time_varying_categorical_variables_decoder]
            + [continuous_vectors[i] for i in self.time_varying_real_variables_decoder],
            dim=2,
        ).transpose(0, 1)[self.encode_length :]
        (embeddings_varying_encoder, encoder_sparse_weights,) = self.encoder_variable_selection(
            embeddings_varying_encoder, static_context_variable_selection[: self.encode_length],
        )

        (embeddings_varying_decoder, decoder_sparse_weights,) = self.decoder_variable_selection(
            embeddings_varying_decoder, static_context_variable_selection[self.encode_length :],
        )
        # LSTM
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder,
            (
                self.static_context_initial_hidden_lstm(static_embedding).expand(self.lstm_layers, -1, -1),
                self.static_context_initial_cell_lstm(static_embedding).expand(self.lstm_layers, -1, -1),
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

        output = self.pos_wise_ff(attn_output[self.encode_length :])

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[self.encode_length :])
        output = self.output_layer(output.transpose(0, 1))

        return output
