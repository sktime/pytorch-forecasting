"""
Time Series Transformer with eXogenous variables (TimeXer)
----------------------------------------------------------
"""

################################################################
# NOTE: This implementation of TimeXer derives from PR #1797.  #
# It is experimental and seeks to clarify design decisions.    #
# IT IS STRICTLY A PART OF THE v2 design of PTF.               #
################################################################

from typing import Any, Optional, Union
import warnings as warn

import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from pytorch_forecasting.metrics import MAE, MAPE, MultiHorizonMetric, QuantileLoss
from pytorch_forecasting.metrics.base_metrics import MultiLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class TimeXer(TslibBaseModel):
    def __init__(
        self,
        loss: nn.Module,
        features: str = "MS",
        enc_in: int = None,
        hidden_size: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        patch_length: int = 24,
        factor: int = 5,
        activation: str = "relu",
        endogenous_vars: Optional[list[str]] = None,
        exogenous_vars: Optional[list[str]] = None,
        logging_metrics: Optional[list[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
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

        self.features = features
        self.enc_in = enc_in
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.patch_length = patch_length
        self.activation = activation
        self.factor = factor
        self.endogenous_vars = endogenous_vars
        self.exogenous_vars = exogenous_vars
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """
        Initialize the network for TimeXer's architecture.
        """

        from pytorch_forecasting.layers.attention._attention_layer import (
            AttentionLayer,
            FullAttention,
        )
        from pytorch_forecasting.layers.embeddings import (
            DataEmbedding_inverted,
            EnEmbedding,
        )
        from pytorch_forecasting.layers.encoders import Encoder, EncoderLayer
        from pytorch_forecasting.layers.output._flatten_head import FlattenHead

        if self.context_length <= self.patch_length:
            raise ValueError(
                f"Context length ({self.context_length}) must be greater than patch"
                "length. Patches of ({self.patch_length}) will end up being longer than"
                "the sequence length."
            )

        if self.context_length % self.patch_length != 0:
            warn.warn(
                f"Context length ({self.context_length}) is not divisible by"
                " patch length. This may lead to unexpected behavior, as some"
                "time steps will not be used in the model."
            )

        self.patch_num = max(1, int(self.context_length // self.patch_length))

        if self.target_dim > 1 and self.features == "M":
            self.n_target_vars = self.target_dim
        else:
            self.n_target_vars = 1

        # currently enc_in is set only to cont_dim since
        # the data module doesn't fully support categorical
        # variables in the context length and modele expects
        # float values.
        self.enc_in = self.enc_in or self.cont_dim

        self.n_quantiles = None

        if hasattr(self.loss, "quantiles"):
            self.n_quantiles = len(self.loss.quantiles)

        if self.hidden_size % self.n_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by n_heads ({self.n_heads}) "  # noqa: E501
                f"for multi-head attention mechanism to work properly."
            )

        self.en_embedding = EnEmbedding(
            self.n_target_vars, self.hidden_size, self.patch_length, self.dropout
        )

        self.ex_embedding = DataEmbedding_inverted(
            self.context_length, self.hidden_size, self.dropout
        )

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
                        self.hidden_size,
                        self.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
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
                    dropout=self.dropout,
                    activation=self.activation,
                )
            )

        self.encoder = Encoder(
            encoder_layers, norm_layer=torch.nn.LayerNorm(self.hidden_size)
        )

        # Initialize output head
        self.head_nf = self.hidden_size * (self.patch_num + 1)
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

    def _forecast(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the TimeXer model.
        Args:
            x (dict[str, torch.Tensor]): Input data.
        Returns:
            dict[str, torch.Tensor]: Model predictions.
        """
        batch_size = x["history_cont"].shape[0]
        history_cont = x["history_cont"]
        history_time_idx = x.get("history_time_idx", None)

        history_target = x.get(
            "history_target",
            torch.zeros(batch_size, self.context_length, 0, device=self.device),
        )  # noqa: E501

        if history_time_idx is not None and history_time_idx.dim() == 2:
            # change [batch_size, time_steps] to [batch_size, time_steps, features]
            history_time_idx = history_time_idx.unsqueeze(-1)

        # explicitly set endogenous and exogenous variables
        endogenous_cont = history_target
        if self.endogenous_vars:
            endogenous_indices = [
                self.cont_names.index(var) for var in self.endogenous_vars
            ]
            endogenous_cont = history_cont[..., endogenous_indices]

        exogenous_cont = history_cont
        if self.exogenous_vars:
            exogenous_indices = [
                self.cont_names.index(var) for var in self.exogenous_vars
            ]
            exogenous_cont = history_cont[..., exogenous_indices]

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

    def _forecast_multi(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forecast for multivariate with multiple time series.

        Args:
            x (dict[str, torch.Tensor]): Input data.
        Returns:
            dict[str, torch.Tensor]: Model predictions.
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

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the TimeXer model.
        Args:
            x (dict[str, torch.Tensor]): Input data.
        Returns:
            dict[str, torch.Tensor]: Model predictions.
        """
        if self.features == "MS":
            out = self._forecast(x)
        else:
            out = self._forecast_multi(x)

        prediction = out[:, : self.prediction_length, :]

        # check to see if the output shape is equal to number of targets
        if prediction.size(2) != self.target_dim:
            prediction = prediction[:, :, : self.target_dim]

        target_indices = range(self.target_dim)
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
