from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class iTransformer(TslibBaseModel):
    """
    An implementation of iTransformer model for v2 of pytorch-forecasting.

    Parameters
    ----------

    References
    ----------
    [1] https://arxiv.org/pdf/2310.06625
    [2] https://github.com/thuml/iTransformer/blob/main/model/iTransformer.py

    Notes
    -----

    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.itransformer._itransformer_pkg_v2 import (
            iTransformer_pkg_v2,
        )

        return iTransformer_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        output_attention: bool = False,
        use_norm: bool = False,
        factor: int = 5,
        d_model: int = 512,
        d_ff: int = 2048,
        activation: str = "relu",
        dropout: float = 0.1,
        n_heads: int = 8,
        e_layers: int = 3,
        logging_metrics: Optional[list[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
        **kwargs,
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

        self.output_attention = output_attention
        self.use_norm = use_norm
        self.factor = factor
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.dropout = dropout
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.freq = self.metadata.get("freq", "h")

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """
        Initialize the network for iTransformer's architecture.
        """
        from pytorch_forecasting.models.itransformer.submodules import (
            AttentionLayer,
            DataEmbedding_inverted,
            Encoder,
            EncoderLayer,
            FullAttention,
        )

        self.enc_embedding = DataEmbedding_inverted(
            self.context_length, self.d_model, self.dropout
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=self.output_attention,
                        ),
                        self.d_model,
                        self.n_heads,
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
        )
        self.projector = nn.Linear(self.d_model, self.prediction_length, bias=True)

    def _forecast(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the iTransformer model.
        Args:
            x (dict[str, torch.Tensor]): Input data.
            Returns:
                dict[str, torch.Tensor]: Model predictions.
        """
        x_enc = x["history_target"]
        x_mark_enc = x["history_cont"]

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # Embedding
        # B L N -> B N E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp)
        # B N E -> B N E
        # the dimensions of embedded time series has been inverted
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[
            :, :, :N
        ]  # filter covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.prediction_length, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.prediction_length, 1)
            )

        return dec_out, attns

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the iTransformer model.
        Args:
            x (dict[str, torch.Tensor]): Input data.
        Returns:
            dict[str, torch.Tensor]: Model predictions.
        """
        dec_out, attns = self._forecast(x)
        prediction = dec_out[:, -self.prediction_length :, :]
        # if prediction.shape[-1] == 1:
        #     prediction = prediction.squeeze(-1)

        if "target_scale" in x:
            prediction = self.transform_output(prediction, x["target_scale"])
        if self.output_attention:
            return {"prediction": prediction, "attention_weights": attns}
        else:
            return {"prediction": prediction}
