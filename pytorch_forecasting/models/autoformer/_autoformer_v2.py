"""
Autoformer model for Pytorch Forecasting.
"""

from typing import Any, Optional, Union
import warnings

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers import (
    AutoCorrelation,
    AutoCorrelationLayer,
    AutoformerDecoder,
    AutoformerDecoderLayer,
    AutoformerEncoder,
    AutoformerEncoderLayer,
    DataEmbedding_wo_pos,
    SeasonalLayerNorm,
    SeriesDecomposition,
)
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class Autoformer(TslibBaseModel):
    """
    Autoformer: Decomposition Transformers with Auto-Correlation
    for Long-Term Time Series Forecasting.

    Autoformer designs a series-wise decomposition connection to
    progressively decompose time series into trend and seasonal components,
    and proposes an Auto-Correlation mechanism to replace standard self-attention.

    Parameters
    ----------
    loss : nn.Module
        Loss function for training step.
    hidden_size : int, default=128
        Dimension of the model embeddings and hidden representations.
    n_heads : int, default=8
        Number of attention heads.
    e_layers : int, default=2
        Number of encoder layers.
    d_layers : int, default=1
        Number of decoder layers.
    d_ff : int, default=512
        Dimension of the feed-forward network.
    moving_avg : int, default=25
        Kernel size for moving average decomposition.
    dropout : float, default=0.1
        Dropout rate.
    factor : int, default=1
        Factor for the Auto-Correlation mechanism.
    activation : str, default='relu'
        Activation function ('relu' or 'gelu').
    embed : str, default='fixed'
        Embedding type ('fixed' or 'learned').
    freq : str, default='h'
        Frequency of the time series.
    label_len : int, default=None
        Length of the overlap history fed into the decoder.
        If None, defaults to context_length // 2.
    logging_metrics : Optional[list[nn.Module]], default=None
        List of metrics to log during training, validation, and testing.
    optimizer : Optional[Union[Optimizer, str]], default='adam'
        Optimizer to use for training.
    optimizer_params : Optional[dict], default=None
        Parameters for the optimizer.
    lr_scheduler : Optional[str], default=None
        Learning rate scheduler to use.
    lr_scheduler_params : Optional[dict], default=None
        Parameters for the learning rate scheduler.
    metadata : Optional[dict], default=None
        Metadata for the model from TslibDataModule.

    References
    ----------
    [1] https://arxiv.org/abs/2106.13008
    [2] https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.autoformer._autoformer_pkg_v2 import (
            Autoformer_pkg_v2,
        )

        return Autoformer_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        hidden_size: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 512,
        moving_avg: int = 25,
        dropout: float = 0.1,
        factor: int = 1,
        activation: str = "relu",
        embed: str = "fixed",
        freq: str = "h",
        label_len: int = None,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            metadata=metadata,
        )

        warnings.warn(
            "Autoformer is an experimental model implemented on TslibBaseModelV2. "
            "It is an unstable version and may be subject to unannounced changes. "
            "Please use with caution."
        )

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.dropout = dropout
        self.factor = factor
        self.activation = activation
        self.embed = embed
        self.freq = freq
        self.label_len = label_len or (self.context_length // 2)

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """
        Initialise the Autoformer model network components.
        """
        # enc_in is target_dim + cont_dim (continuous exogenous features + targets)
        self.enc_in = self.cont_dim + self.target_dim

        self.n_quantiles = None
        if isinstance(self.loss, QuantileLoss):
            self.n_quantiles = len(self.loss.quantiles)

        c_out = self.target_dim
        if self.n_quantiles is not None:
            c_out = self.target_dim * self.n_quantiles

        # Decomp
        self.decomp = SeriesDecomposition(self.moving_avg)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            self.enc_in, self.hidden_size, self.embed, self.freq, self.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            self.target_dim, self.hidden_size, self.embed, self.freq, self.dropout
        )

        # Encoder
        self.encoder = AutoformerEncoder(
            [
                AutoformerEncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        self.hidden_size,
                        self.n_heads,
                    ),
                    self.hidden_size,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.e_layers)
            ],
            norm_layer=SeasonalLayerNorm(self.hidden_size),
        )

        # Decoder
        self.decoder = AutoformerDecoder(
            [
                AutoformerDecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        self.hidden_size,
                        self.n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        self.hidden_size,
                        self.n_heads,
                    ),
                    self.hidden_size,
                    c_out,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.d_layers)
            ],
            norm_layer=SeasonalLayerNorm(self.hidden_size),
            projection=nn.Linear(self.hidden_size, c_out, bias=True),
        )

    def _forecast(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Core forecast method of Autoformer.
        """
        x_enc, target_indices = self._prepare_input_data(x)
        # x_enc has shape (batch_size, context_length, enc_in)
        batch_size = x_enc.shape[0]

        # Extract target channels for decoder initialization
        x_enc_target = x_enc[:, :, target_indices]

        # decomp init
        mean = torch.mean(x_enc_target, dim=1, keepdim=True).repeat(
            1, self.prediction_length, 1
        )
        zeros = torch.zeros(
            [batch_size, self.prediction_length, self.target_dim],
            dtype=x_enc.dtype,
            device=x_enc.device,
        )
        seasonal_init, trend_init = self.decomp(x_enc_target)

        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len :, :], zeros], dim=1
        )

        # If n_quantiles is not None, repeat trend_init along channels to match c_out
        if self.n_quantiles is not None:
            trend_init = trend_init.repeat(1, 1, self.n_quantiles)

        # x_mark can be None as DataEmbedding_wo_pos gracefully falls back
        x_mark_enc = None
        x_mark_dec = None

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )

        # final
        dec_out = trend_part + seasonal_part
        # extract prediction length from the end of sequence
        prediction = dec_out[:, -self.prediction_length :, :]

        if self.n_quantiles is not None:
            # Reshape prediction from (batch, pred_len, target_dim * n_quantiles)
            # to (batch, pred_len, n_quantiles) for target_dim=1.
            if self.target_dim == 1:
                prediction = prediction.reshape(
                    batch_size, self.prediction_length, self.n_quantiles
                )
            else:
                prediction = prediction.reshape(
                    batch_size,
                    self.prediction_length,
                    self.target_dim,
                    self.n_quantiles,
                )

        return prediction

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the Autoformer model.
        """
        prediction = self._forecast(x)

        if "target_scale" in x and hasattr(self, "transform_output"):
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}

    def _prepare_input_data(self, x: dict[str, torch.Tensor]):
        """Prepare input data and target indices for model input."""
        available_features = []
        target_indices = []
        current_idx = 0

        if "history_cont" in x and x["history_cont"].size(-1) > 0:
            available_features.append(x["history_cont"])
            current_idx += x["history_cont"].size(-1)

        if "history_target" in x and x["history_target"].size(-1) > 0:
            n_targets = x["history_target"].size(-1)
            target_indices = list(range(current_idx, current_idx + n_targets))
            available_features.append(x["history_target"])
        elif "history_target" in x:
            # history_target might be 2D (batch, context_length), shape must be 3D
            target_3d = x["history_target"]
            if target_3d.dim() == 2:
                target_3d = target_3d.unsqueeze(-1)
            n_targets = target_3d.size(-1)
            target_indices = list(range(current_idx, current_idx + n_targets))
            available_features.append(target_3d)

        if not available_features:
            raise ValueError("No valid input features found in the input dictionary.")

        input_data = torch.cat(available_features, dim=-1)

        target_indices = (
            torch.tensor(target_indices, dtype=torch.long, device=input_data.device)
            if target_indices
            else None
        )

        return input_data, target_indices
