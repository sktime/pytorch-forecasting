"""
Time Series Transformer with eXogenous variables (TimeXer)
----------------------------------------------------------
"""

################################################################
# NOTE: This implementation of TimeXer derives from PR #1797.  #
# It is experimental and seeks to clarify design decisions.    #
################################################################


from typing import Any, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.metrics import MAE, MAPE, Metric, QuantileLoss
from pytorch_forecasting.metrics.base_metrics import MultiLoss
from pytorch_forecasting.models.base._base_model_v2 import TslibBaseModel


class TimeXer(TslibBaseModel):
    def __init__(
        self,
        loss: Metric,
        context_length: int,
        prediction_length: int,
        features: str = "MS",
        enc_in: int = None,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        patch_length: int = 24,
        factor: int = 5,
        endogenous_vars: Optional[List[str]] = None,
        exogenous_vars: Optional[List[str]] = None,
        logging_metrics: Optional[List[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[Dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
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

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.endogenous_vars = endogenous_vars
        self.exogenous_vars = exogenous_vars
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

    def _init_network(self):
        """
        Initialize the network for TimeXer's architecture.
        """

        from pytorch_forecasting.layers.attention import (
            AttentionLayer,
            FullAttention,
        )
        from pytorch_forecasting.layers.embeddings import (
            DataEmbedding_inverted,
            EnEmbedding,
            PositionalEmbedding,
        )
        from pytorch_forecasting.layers.encoders import Encoder, EncoderLayer
        from pytorch_forecasting.layers.output_layers import FlattenHead

        self.patch_num = max(1, int(self.context_length // self.patch_length))

        if self.target_dim > 1 and self.features == "M":
            self.n_target_vars = self.target_dim
        else:
            self.n_target_vars = 1

        self.enc_in = self.enc_in or self.cont_dim

        self.n_quantiles = None

        if hasattr(self, "quantiles"):
            self.n_quantiles = len(self.loss.quantiles)

        self.en_embedding = EnEmbedding(
            self.n_target_vars, self.d_model, self.patch_length, self.dropout
        )

        self.ex_embedding = DataEmbedding_inverted(
            self.context_length, self.d_model, self.dropout
        )

        encoder_layers = []

        encoder_layers = []
        for _ in range(self.e_layers):
            encoder_layers.append(
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        self.d_model,
                        self.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
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
            )

        self.encoder = Encoder(
            encoder_layers, norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        # Initialize output head
        self.head_nf = self.d_model * (self.patch_num + 1)
        self.head = FlattenHead(
            self.enc_in,
            self.head_nf,
            self.prediction_length,
            head_dropout=self.dropout,
            n_quantiles=self.n_quantiles,
        )

    # def _get_target_positions(self) -> torch.Tensor:
    #     """
    #     Get the target positions from the dataset.
    #     Returns:
    #         torch.Tensor: Target positions.
    #     """
    #     return torch.tensor(self.target_indices, device=self.device)

    def _forecast(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the TimeXer model.
        Args:
            x (Dict[str, torch.Tensor]): Input data.
        Returns:
            Dict[str, torch.Tensor]: Model predictions.
        """
        batch_size = x["history_cont"].shape[0]
        history_cont = x["history_cont"]
        history_time_idx = x.get("history_time_idx", None)

        history_target = x.get(
            "history_target",
            torch.zeros(batch_size, self.context_length, 0, device=self.device),
        )  # noqa: E501
        if self.endogenous_vars:
            endogenous_indices = [
                self.cont_names.index(var) for var in self.endogenous_vars
            ]
            endogenous_cont = history_cont[..., endogenous_indices]
        else:
            endogenous_cont = history_target

        if self.exogenous_vars:
            exogenous_indices = [
                self.cont_names.index(var) for var in self.exogenous_vars
            ]
            exogenous_cont = history_cont[..., exogenous_indices]
        else:
            exogenous_cont = history_cont

        en_embed, n_vars = self.en_embedding(endogenous_cont)
        ex_embed = self.ex_embedding(exogenous_cont, history_time_idx)

        enc_out = self.encoder(en_embed, ex_embed)

        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )

        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)

        if self.n_quantiles is not None:
            dec_out = dec_out.permute(0, 2, 1, 3)
        else:
            dec_out = dec_out.permute(0, 2, 1)

        return dec_out

    def _forecast_multi(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forecast for multivariate with multiple time series.

        Args:
            x (Dict[str, torch.Tensor]): Input data.
        Returns:
            Dict[str, torch.Tensor]: Model predictions.
        """

        history_cont = x["history_cont"]
        history_time_idx = x.get("history_time_idx", None)
        history_target = x["history_target"]

        if self.endogenous_vars:
            endogenous_indices = [
                self.cont_names.index(var) for var in self.endogenous_vars
            ]
            endogenous_cont = history_cont[..., endogenous_indices]
        else:
            endogenous_cont = history_target

        if self.exogenous_vars:
            exogenous_indices = [
                self.cont_names.index(var) for var in self.exogenous_vars
            ]
            exogenous_cont = history_cont[..., exogenous_indices]
        else:
            exogenous_cont = history_cont

        en_embed, n_vars = self.en_embedding(endogenous_cont)

        ex_embed = self.ex_embedding(exogenous_cont, history_time_idx)

        enc_out = self.encoder(en_embed, ex_embed)

        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )

        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)

        if self.n_quantiles is not None:
            dec_out = dec_out.permute(0, 2, 1, 3)
        else:
            dec_out = dec_out.permute(0, 2, 1)

        return dec_out

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the TimeXer model.
        Args:
            x (Dict[str, torch.Tensor]): Input data.
        Returns:
            Dict[str, torch.Tensor]: Model predictions.
        """
        if self.features == "MS":
            out = self._forecast(x)
        else:
            out = self._forecast_multi(x)

        prediction = out[:, : self.prediction_length, :]

        # check to see if the output shape is equal to number of targets
        if prediction.size(2) != self.target_dim:
            prediction = prediction[:, :, : self.target_dim]

        target_indices = range(len(self.target_dim))
        if self.n_quantiles is not None:
            if self.target_dim > 1:
                prediction = [prediction[..., i, :] for i in target_indices]
            else:
                prediction = prediction[..., 0]
        else:
            if len(target_indices) == 1:
                prediction = prediction[..., 0]
            else:
                prediction = [prediction[..., i] for i in target_indices]

        if "target_scale" in x:
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
