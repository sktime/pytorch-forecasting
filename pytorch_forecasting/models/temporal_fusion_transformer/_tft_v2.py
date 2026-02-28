########################################################################################
# Disclaimer: This implementation is based on the new version of data pipeline and is
# experimental, please use with care.
#
# This v2 TFT uses the original TFT paper architecture (GatedResidualNetwork,
# VariableSelectionNetwork, InterpretableMultiHeadAttention, etc.) while relying
# on the v2 metadata-based data pipeline - no data processing logic lives here.
########################################################################################

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_model_v2 import BaseModel
from pytorch_forecasting.models.nn import LSTM
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    AddNorm,
    GateAddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
)


class TFT(BaseModel):
    """Temporal Fusion Transformer (v2) with original TFT architecture.

    Uses the v2 metadata-based data pipeline (no data processing in the model).
    Implements the architecture from the TFT paper using the custom sub-modules:

    * GatedResidualNetwork for non-linear processing with gating
    * VariableSelectionNetwork for instance-wise variable importance
    * InterpretableMultiHeadAttention for long-range temporal patterns
    * Gated skip connections (GatedLinearUnit + AddNorm / GateAddNorm)
    * Static context enrichment via dedicated GRNs

    Parameters
    ----------
    loss : nn.Module
        Loss function for training.
    logging_metrics : list[nn.Module] or None
        Metrics to log during training.
    optimizer : Optimizer or str or None
        Optimizer ("adam", "sgd", or Optimizer instance).
    optimizer_params : dict or None
        Optimizer keyword arguments.
    lr_scheduler : str or None
        Learning rate scheduler name.
    lr_scheduler_params : dict or None
        Scheduler keyword arguments.
    hidden_size : int
        Main hidden size of the network (controls all internal dimensions).
    lstm_layers : int
        Number of LSTM layers.
    attention_head_size : int
        Number of attention heads. hidden_size must be divisible by this.
    dropout : float
        Dropout rate used throughout the network.
    hidden_continuous_size : int
        Size of the linear projection (prescaler) for each raw continuous variable.
        Each 1D scalar from the data pipeline is stretched to this size via
        ``nn.Linear(1, hidden_continuous_size)`` before being fed into the
        per-variable GRN inside the VariableSelectionNetwork.
    metadata : dict or None
        Metadata dict from the data module.
    output_size : int or list of int
        Number of outputs (e.g. number of quantiles for QuantileLoss).
    causal_attention : bool
        Whether to use causal masking in the decoder self-attention.
    mask_bias : float
        Bias value for masked positions in attention (-1e9 or -inf).
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.temporal_fusion_transformer._tft_pkg_v2 import (
            TFT_pkg_v2,
        )

        return TFT_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        hidden_size: int = 64,
        lstm_layers: int = 1,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        metadata: dict | None = None,
        output_size: int | list[int] = 1,
        share_single_variable_networks: bool = False,
        causal_attention: bool = True,
        mask_bias: float = -1e9,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.metadata = metadata or {}
        self.output_size = output_size
        self.hidden_continuous_size = hidden_continuous_size
        self.share_single_variable_networks = share_single_variable_networks
        self.causal_attention = causal_attention
        self.mask_bias = mask_bias

        self.max_encoder_length = self.metadata.get("max_encoder_length", 10)
        self.max_prediction_length = self.metadata.get("max_prediction_length", 1)
        self.encoder_cont = self.metadata.get("encoder_cont", 0)
        self.encoder_cat = self.metadata.get("encoder_cat", 0)
        self.encoder_input_dim = self.encoder_cont + self.encoder_cat
        self.decoder_cont = self.metadata.get("decoder_cont", 0)
        self.decoder_cat = self.metadata.get("decoder_cat", 0)
        self.decoder_input_dim = self.decoder_cont + self.decoder_cat
        self.static_cont_dim = self.metadata.get("static_continuous_features", 0)
        self.static_cat_dim = self.metadata.get("static_categorical_features", 0)
        self.static_input_dim = self.static_cont_dim + self.static_cat_dim

        # Synthetic variable names for VariableSelectionNetwork since
        # new datapipeline outputs 1d tensor.
        """This is how metadata looks: {'encoder_cat': 2,
            'encoder_cont': 3,
            'decoder_cat': 0,
            'decoder_cont': 1,
            'target': 1,
            'static_categorical_features': 1,
            'static_continuous_features': 1,
            'max_encoder_length': 30,
            'max_prediction_length': 1,
            'min_encoder_length': 30,
            'min_prediction_length': 1}"""

        self._enc_var_names = [f"enc_cont_{i}" for i in range(self.encoder_cont)] + [
            f"enc_cat_{i}" for i in range(self.encoder_cat)
        ]
        self._dec_var_names = [f"dec_cont_{i}" for i in range(self.decoder_cont)] + [
            f"dec_cat_{i}" for i in range(self.decoder_cat)
        ]
        self._static_var_names = [
            f"static_cont_{i}" for i in range(self.static_cont_dim)
        ] + [f"static_cat_{i}" for i in range(self.static_cat_dim)]

        # Shared prescalers to stretch raw 1D inputs before feeding into VSN.
        # Continuous vars: nn.Linear(1, hidden_continuous_size)
        # Categorical vars: nn.Linear(1, hidden_size) — acts as a learned
        #   pseudo-embedding since the v2 data pipeline only provides
        #   label-encoded integers (dim 1), not dense embedding vectors.
        #   TODO: replace with proper nn.Embedding once metadata includes
        #   per-variable cardinality. The new pipeline does not currently
        #   return the vocabulary size for categorical variables.

        all_continuous_names = (
            [f"static_cont_{i}" for i in range(self.static_cont_dim)]
            + [f"enc_cont_{i}" for i in range(self.encoder_cont)]
            + [f"dec_cont_{i}" for i in range(self.decoder_cont)]
        )
        all_categorical_names = (
            [f"static_cat_{i}" for i in range(self.static_cat_dim)]
            + [f"enc_cat_{i}" for i in range(self.encoder_cat)]
            + [f"dec_cat_{i}" for i in range(self.decoder_cat)]
        )
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(1, self.hidden_continuous_size)
                for name in all_continuous_names
            }
        )
        self.prescalers.update(
            {name: nn.Linear(1, self.hidden_size) for name in all_categorical_names}
        )

        enc_input_sizes = {
            f"enc_cat_{i}": self.hidden_size for i in range(self.encoder_cat)
        }
        enc_input_sizes.update(
            {
                f"enc_cont_{i}": self.hidden_continuous_size
                for i in range(self.encoder_cont)
            }
        )

        dec_input_sizes = {
            f"dec_cat_{i}": self.hidden_size for i in range(self.decoder_cat)
        }
        dec_input_sizes.update(
            {
                f"dec_cont_{i}": self.hidden_continuous_size
                for i in range(self.decoder_cont)
            }
        )

        # THE SHARED GRN LOGIC
        self.shared_single_variable_grns = {}
        if self.share_single_variable_networks:
            for i in range(self.encoder_cont):
                grn = GatedResidualNetwork(
                    self.hidden_continuous_size,
                    min(self.hidden_continuous_size, self.hidden_size),
                    self.hidden_size,
                    self.dropout,
                )
                self.shared_single_variable_grns[f"enc_cont_{i}"] = grn
                if i < self.decoder_cont:
                    self.shared_single_variable_grns[f"dec_cont_{i}"] = grn
            for i in range(self.encoder_cat):
                grn = GatedResidualNetwork(
                    self.hidden_size,
                    self.hidden_size,
                    self.hidden_size,
                    self.dropout,
                )
                self.shared_single_variable_grns[f"enc_cat_{i}"] = grn
                if i < self.decoder_cat:
                    self.shared_single_variable_grns[f"dec_cat_{i}"] = grn

        # 1. Static Variable Selection
        if self.static_input_dim > 0:
            static_input_sizes = {
                f"static_cat_{i}": self.hidden_size for i in range(self.static_cat_dim)
            }
            static_input_sizes.update(
                {
                    f"static_cont_{i}": self.hidden_continuous_size
                    for i in range(self.static_cont_dim)
                }
            )
            self.static_variable_selection = VariableSelectionNetwork(
                input_sizes=static_input_sizes,
                hidden_size=self.hidden_size,
                input_embedding_flags={
                    f"static_cat_{i}": True for i in range(self.static_cat_dim)
                },
                dropout=self.dropout,
                prescalers=self.prescalers,
            )
        else:
            self.static_variable_selection = None

        # 2. Encoder Variable Selection
        if self.encoder_input_dim > 0:
            self.encoder_variable_selection = VariableSelectionNetwork(
                input_sizes=enc_input_sizes,
                hidden_size=self.hidden_size,
                input_embedding_flags={
                    f"enc_cat_{i}": True for i in range(self.encoder_cat)
                },
                dropout=self.dropout,
                context_size=self.hidden_size,
                prescalers=self.prescalers,
                single_variable_grns=self.shared_single_variable_grns,
            )
        else:
            self.encoder_variable_selection = None

        # 3. Decoder Variable Selection
        if self.decoder_input_dim > 0:
            self.decoder_variable_selection = VariableSelectionNetwork(
                input_sizes=dec_input_sizes,
                hidden_size=self.hidden_size,
                input_embedding_flags={
                    f"dec_cat_{i}": True for i in range(self.decoder_cat)
                },
                dropout=self.dropout,
                context_size=self.hidden_size,
                prescalers=self.prescalers,
                single_variable_grns=self.shared_single_variable_grns,
            )
        else:
            self.decoder_variable_selection = None

        # 4. Static Context GRNs
        # (a) context for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )
        # (b) initial hidden state for LSTM
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )
        # (c) initial cell state for LSTM
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )
        # (d) context for post-LSTM static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # 5. LSTM Encoder & Decoder
        self.lstm_encoder = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )
        self.lstm_decoder = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )

        # 6. Post-LSTM Gating (skip connections over LSTM)
        self.post_lstm_gate_encoder = GatedLinearUnit(hidden_size, dropout=dropout)
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        self.post_lstm_add_norm_encoder = AddNorm(hidden_size, trainable_add=False)
        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # 7. Static Enrichment (after LSTM, before attention)
        self.static_enrichment = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size,
        )

        # 8. Interpretable Multi-Head Attention
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=hidden_size,
            n_head=attention_head_size,
            dropout=dropout,
            mask_bias=mask_bias,
        )
        self.post_attn_gate_norm = GateAddNorm(
            hidden_size, dropout=dropout, trainable_add=False
        )

        # 9. Position-wise Feed-Forward
        self.pos_wise_ff = GatedResidualNetwork(
            hidden_size,
            hidden_size,
            hidden_size,
            dropout=dropout,
        )

        # 10. Output Processing
        self.pre_output_gate_norm = GateAddNorm(
            hidden_size, dropout=None, trainable_add=False
        )
        if isinstance(output_size, list):
            self.output_layer = nn.ModuleList(
                [nn.Linear(hidden_size, os) for os in output_size]
            )
        else:
            self.output_layer = nn.Linear(hidden_size, output_size)

    # Helper methods

    def _expand_static_context(self, context, timesteps):
        """Add time dimension to static context."""
        return context[:, None].expand(-1, timesteps, -1)

    def _get_attention_mask(self, batch_size):
        """Build attention mask for InterpretableMultiHeadAttention.

        Shape: (batch, decoder_len, encoder_len + decoder_len).
        True positions are NOT attended to.
        """
        enc_len = self.max_encoder_length
        dec_len = self.max_prediction_length

        if self.causal_attention:
            attend_step = torch.arange(dec_len, device=self.device)
            predict_step = torch.arange(dec_len, device=self.device)[:, None]
            decoder_mask = attend_step >= predict_step
        else:
            decoder_mask = torch.zeros(
                dec_len, dec_len, dtype=torch.bool, device=self.device
            )

        encoder_mask = torch.zeros(
            dec_len, enc_len, dtype=torch.bool, device=self.device
        )

        mask = torch.cat([encoder_mask, decoder_mask], dim=1)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        return mask

    def _split_to_variable_dict(
        self, cont, cat, cont_names, cat_names, squeeze_time=False
    ):
        """Split flat feature tensors into per-variable dicts for VSN."""
        result = {}
        if cont is not None and len(cont_names) > 0:
            cont = cont.to(dtype=torch.float32)
            for i, name in enumerate(cont_names):
                var = cont[..., i : i + 1]
                if squeeze_time and var.ndim == 3:
                    var = var[:, 0, :]
                result[name] = var
        if cat is not None and len(cat_names) > 0:
            cat = cat.to(dtype=torch.float32)
            for i, name in enumerate(cat_names):
                var = cat[..., i : i + 1]
                if squeeze_time and var.ndim == 3:
                    var = var[:, 0, :]
                result[name] = var
        return result

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass implementing the full TFT architecture.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Batch dictionary from EncoderDecoderTimeSeriesDataModule.

        Returns
        -------
        dict[str, torch.Tensor]
            prediction tensor of shape (batch, prediction_length, output_size).
        """
        batch_size = x["encoder_cont"].shape[0]
        timesteps = self.max_encoder_length + self.max_prediction_length

        # Raw tensors from batch
        encoder_cont = x.get("encoder_cont")
        encoder_cat = x.get("encoder_cat")
        decoder_cont = x.get("decoder_cont")
        decoder_cat = x.get("decoder_cat")
        static_cont = x.get("static_continuous_features")
        static_cat = x.get("static_categorical_features")

        # 1. Static Variable Selection
        if self.static_variable_selection is not None and self.static_input_dim > 0:
            static_dict = self._split_to_variable_dict(
                static_cont,
                static_cat,
                [f"static_cont_{i}" for i in range(self.static_cont_dim)],
                [f"static_cat_{i}" for i in range(self.static_cat_dim)],
                squeeze_time=True,
            )
            static_embedding, static_variable_selection_weights = (
                self.static_variable_selection(static_dict)
            )
        else:
            static_embedding = torch.zeros(
                batch_size,
                self.hidden_size,
                dtype=torch.float32,
                device=self.device,
            )

        # 2. Static Context GRNs
        static_context_variable_selection = self._expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )
        static_context_initial_hidden = self.static_context_initial_hidden_lstm(
            static_embedding
        )
        static_context_initial_cell = self.static_context_initial_cell_lstm(
            static_embedding
        )
        static_context_enrichment = self.static_context_enrichment(static_embedding)

        # 3. Encoder Variable Selection
        if self.encoder_variable_selection is not None and self.encoder_input_dim > 0:
            encoder_dict = self._split_to_variable_dict(
                encoder_cont,
                encoder_cat,
                [f"enc_cont_{i}" for i in range(self.encoder_cont)],
                [f"enc_cat_{i}" for i in range(self.encoder_cat)],
            )
            embeddings_varying_encoder, encoder_sparse_weights = (
                self.encoder_variable_selection(
                    encoder_dict,
                    static_context_variable_selection[:, : self.max_encoder_length],
                )
            )
        else:
            embeddings_varying_encoder = torch.zeros(
                batch_size,
                self.max_encoder_length,
                self.hidden_size,
                dtype=torch.float32,
                device=self.device,
            )

        # 4. Decoder Variable Selection
        if self.decoder_variable_selection is not None and self.decoder_input_dim > 0:
            decoder_dict = self._split_to_variable_dict(
                decoder_cont,
                decoder_cat,
                [f"dec_cont_{i}" for i in range(self.decoder_cont)],
                [f"dec_cat_{i}" for i in range(self.decoder_cat)],
            )
            embeddings_varying_decoder, decoder_sparse_weights = (
                self.decoder_variable_selection(
                    decoder_dict,
                    static_context_variable_selection[:, self.max_encoder_length :],
                )
            )
        else:
            embeddings_varying_decoder = torch.zeros(
                batch_size,
                self.max_prediction_length,
                self.hidden_size,
                dtype=torch.float32,
                device=self.device,
            )

        # 5. LSTM (initialised from static context)
        input_hidden = static_context_initial_hidden.expand(self.lstm_layers, -1, -1)
        input_cell = static_context_initial_cell.expand(self.lstm_layers, -1, -1)

        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder, (input_hidden, input_cell)
        )
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder, (hidden, cell)
        )

        # 6. Post-LSTM Gating + Skip Connection
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(
            lstm_output_encoder, embeddings_varying_encoder
        )

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(
            lstm_output_decoder, embeddings_varying_decoder
        )

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # 7. Static Enrichment
        attn_input = self.static_enrichment(
            lstm_output,
            self._expand_static_context(static_context_enrichment, timesteps),
        )

        # 8. Interpretable Multi-Head Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, self.max_encoder_length :],
            k=attn_input,
            v=attn_input,
            mask=self._get_attention_mask(batch_size),
        )

        # 9. Post-Attention Gating + Skip
        attn_output = self.post_attn_gate_norm(
            attn_output, attn_input[:, self.max_encoder_length :]
        )

        # 10. Position-wise Feed-Forward
        output = self.pos_wise_ff(attn_output)

        # 11. Pre-Output Gating + Skip (over the temporal fusion decoder)
        output = self.pre_output_gate_norm(
            output, lstm_output[:, self.max_encoder_length :]
        )

        # 12. Output Projection
        if isinstance(self.output_layer, nn.ModuleList):
            prediction = [layer(output) for layer in self.output_layer]
        else:
            prediction = self.output_layer(output)

        return {"prediction": prediction}
