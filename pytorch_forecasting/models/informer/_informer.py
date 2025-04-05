"""
Informer Transformer for Long Sequence Time-Series Forecasting.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric
from pytorch_forecasting.models.base import BaseModel
from pytorch_forecasting.models.informer.sub_modules import (
    AttentionLayer,
    ConvLayer,
    DataEmbedding,
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    ProbAttention,
)
from pytorch_forecasting.utils._dependencies import _check_matplotlib


class Informer(BaseModel):
    def __init__(
        self,
        encoder_input: int,
        decoder_input: int,
        out_channels: int,
        task: str,
        seq_len: int,
        label_len: int,
        out_len: int,
        factor: int = 5,
        d_model: int = 512,
        n_heads: int = 8,
        encoder_layers: Union[int, List[int]] = 3,
        decoder_layers: int = 2,
        d_ff: int = 512,
        dropout: int = 0.0,
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        loss: MultiHorizonMetric = None,
        distil: bool = True,
        mix: bool = True,
        logging_metrics: Optional[nn.ModuleList] = None,
        **kwargs,
    ):
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        if loss is None:
            loss = MAE()
        self.save_hyperparameters()
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)
        self.enc_embedding = DataEmbedding(
            self.encoder_input, self.d_model, self.embed, self.freq, self.dropout
        )
        self.dec_embedding = DataEmbedding(
            self.decoder_input, self.d_model, self.embed, self.freq, self.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        self.d_model,
                        self.n_heads,
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.encoder_layers)
            ],
            (
                [ConvLayer(self.d_model) for l in range(self.encoder_layers - 1)]
                if self.distil and ("forecast" in self.task_name)
                else None
            ),
            norm_layer=torch.nn.LayerNorm(self.d_model),
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            True,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        self.d_model,
                        self.n_heads,
                    ),
                    AttentionLayer(
                        ProbAttention(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        self.d_model,
                        self.n_heads,
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.decoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.out_channels, bias=True),
        )

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        """
        Convenience function to create network from :py:class`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.

        Args:
            dataset (TimeSeriesDataSet): dataset where sole predictor is the target.
            **kwargs: additional arguments to be passed to ``__init__`` method.

        Returns:
            Informer
        """  # noqa: E501
        new_kwargs = {
            "prediction_length": dataset.max_prediction_length,
            "context_length": dataset.max_encoder_length,
        }
        new_kwargs.update(kwargs)

        # create class and return
        return super().from_dataset(
            dataset,
            **new_kwargs,
        )

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        return dec_out  # [B, L, D]

    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out  # [B, L, D]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "long_term_forecast":
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "short_term_forecast":
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        return None
