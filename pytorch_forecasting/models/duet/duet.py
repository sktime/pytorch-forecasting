from copy import copy
from logging import config
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from pytorch_forecasting.utils._dependencies._safe_import import _safe_import

Metric = _safe_import("torchmetrics.Metric")
rearrange = _safe_import("einops.rearrange")

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    RMSE,
    SMAPE,
    MultiHorizonMetric,
    QuantileLoss,
)
from pytorch_forecasting.models.base import BaseModelWithCovariates
from pytorch_forecasting.models.duet.sub_modules.layers.linear_extractor_cluster import (  # noqa: E501
    Linear_extractor_cluster,
)
from pytorch_forecasting.models.duet.sub_modules.utils.masked_attention import (
    AttentionLayer,
    Encoder,
    EncoderLayer,
    FullAttention,
    Mahalanobis_mask,
)
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding

# Default hyperparameters as specified in the official implementation.
DEFAULT_DUET_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 2,
    "d_layers": 1,
    "d_model": 512,
    "d_ff": 2048,
    "hidden_size": 256,
    "freq": "h",
    "factor": 1,
    "n_heads": 8,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "stride": 8,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "moving_avg": 25,
    "batch_size": 256,
    "lradj": "type3",
    "num_workers": 0,
    "patience": 10,
    "num_experts": 4,
    "noisy_gating": True,
    "k": 1,
    "CI": False,
}


class DUETModel(BaseModelWithCovariates):
    """
    Initial implementation of DUET: Dual Clustering Enhanced Multivariate Time
    Series Forecasting

    Original paper: https://arxiv.org/pdf/2412.10859
    """

    def __init__(
        self,
        # Parameters for BaseModelWithCovariates
        static_categoricals: list[str],
        static_reals: list[str],
        time_varying_categoricals_encoder: list[str],
        time_varying_categoricals_decoder: list[str],
        time_varying_reals_encoder: list[str],
        time_varying_reals_decoder: list[str],
        x_reals: list[str],
        x_categoricals: list[str],
        embedding_sizes: dict,
        embedding_labels: dict,
        embedding_paddings: list[str],
        categorical_groups: dict,
        # Parameters for BaseModel
        loss: Metric,
        learning_rate: float,
        dataset_parameters: dict,
        optimizer: str = "adam",
        output_transformer: Callable = None,
        # Parameters for DUET's architecture
        dec_in: int = 1,
        c_out: int = 1,
        e_layers: int = 2,
        d_layers: int = 1,
        d_model: int = 512,
        d_ff: int = 2048,
        hidden_size: int = 256,
        freq: str = "h",
        factor: int = 1,
        n_heads: int = 8,
        activation: str = "gelu",
        output_attention: int = 0,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.2,
        fc_dropout: float = 0.2,
        moving_avg: int = 25,
        lradj: str = "type3",
        num_workers: int = 0,
        patience: int = 10,
        num_experts: int = 4,
        noisy_gating: bool = True,
        k: int = 1,
        CI: bool = False,
        **kwargs,
    ):
        """
        Initialize DUET model.

        Args:
            static_categoricals: names of static categorical variables
            static_reals: names of static continuous variables
            time_varying_categoricals_encoder: names of categorical variables for encoder
            time_varying_categoricals_decoder: names of categorical variables for decoder
            time_varying_reals_encoder: names of continuous variables for encoder
            time_varying_reals_decoder: names of continuous variables for decoder
            x_reals: order of continuous variables in tensor passed to forward function
            x_categoricals: order of categorical variables in tensor passed to forward function
            embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
                embedding size
            embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
            embedding_labels: dictionary mapping (string) indices to list of categorical labels
            categorical_groups: dictionary where values
                are list of categorical variables that are forming together a new categorical
                variable which is the key in the dictionary
            loss: loss function to use. Can be any instance of torchmetrics.Metric
            learning_rate: learning rate to use
            dataset_parameters: dictionary containing parameters of the dataset
            optimizer: optimizer to use (default: "adam")
            output_transformer: function to transform the output of the network
            dec_in: number of input features to the decoder
            c_out: number of output features
            e_layers: number of encoder layers
            d_layers: number of decoder layers
            d_model: dimensionality of the model's hidden states. It is the size of the
                vectors used to represent the time series data throughout the model.
            d_ff: dimensionality of the feedforward network model
            hidden_size: hidden size for the distributional router's encoder,
                which is part of the Mixture of Experts mechanism.
            freq: frequency of the time series data. This is used for generating time
                based features.
            factor: factor for the attention mechanism
            n_heads: number of attention heads
            activation: activation function used in the model
            output_attention: whether to output attention weights
            patch_len: length of each patch when using patching mechanism
            stride: stride for the patching mechanism
            dropout: dropout rate applied within the encoder layers to prevent overfitting
            fc_dropout: dropout rate applied to the final fully connected layer
            moving_avg: window size for moving average, used in the decomposition of the time series
            batch_size: batch size used during training
            lradj: learning rate adjustment strategy
            num_workers: number of workers for data loading
            patience: number of epochs with no improvement after which training will be stopped
            num_experts: number of experts in the Mixture of Experts mechanism
            noisy_gating: whether to use noisy gating in the Mixture of Experts mechanism
            k: number of experts to be selected by the router for each input
            CI: whether to use channel independent configuration. If True, the model
                processes each channel (variate) of the time series independently before combining them.
            **kwargs: additional arguments
        """  # noqa: E501
        self.save_hyperparameters()
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            optimizer=optimizer,
            output_transformer=output_transformer,
            dataset_parameters=dataset_parameters,
        )

        self.cat_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            embedding_paddings=self.hparams.embedding_paddings,
            categorical_groups=self.hparams.categorical_groups,
            x_categoricals=self.hparams.x_categoricals,
        )

        self.cluster = Linear_extractor_cluster(self.hparams)
        self.CI = self.hparams.CI
        self.n_vars = self.hparams.enc_in
        self.mask_generator = Mahalanobis_mask(self.hparams.seq_len)
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            self.hparams.factor,
                            attention_dropout=self.hparams.dropout,
                            output_attention=self.hparams.output_attention,
                        ),
                        self.hparams.d_model,
                        self.hparams.n_heads,
                    ),
                    self.hparams.d_model,
                    self.hparams.d_ff,
                    dropout=self.hparams.dropout,
                    activation=self.hparams.activation,
                )
                for _ in range(self.hparams.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.hparams.d_model),
        )

        self.linear_head = nn.Sequential(
            nn.Linear(self.hparams.d_model, self.hparams.pred_len),
            nn.Dropout(self.hparams.fc_dropout),
        )

    def forward(self, x: dict) -> dict:
        embedded_features = self.cat_embeddings(x["encoder_cat"])
        cont_tensor = x["encoder_cont"]

        input_tensor = torch.cat([cont_tensor, embedded_features["cols"]], dim=-1)

        batch_size = input_tensor.shape[0]  # noqa: F841

        if self.hparams.CI:
            channel_independent_input = rearrange(input_tensor, "b l n -> (b n) l 1")

            reshaped_output, L_importance = self.cluster(channel_independent_input)

            temporal_feature = rearrange(
                reshaped_output, "(b n) l 1 -> b l n", b=input_tensor.shape[0]
            )  # noqa: E501
        else:
            temporal_feature, L_importance = self.cluster(input_tensor)

        temporal_feature = rearrange(temporal_feature, "b d n -> b n d")

        if self.n_vars > 1:
            # Multivariate case: apply channel transformer

            changed_input = rearrange(input_tensor, "b l n -> b n l")

            channel_mask = self.mask_generator(changed_input)

            channel_group_feature, _ = self.Channel_transformer(
                x=temporal_feature, attn_mask=channel_mask
            )
        else:
            # For univariate case, the group feature is just the temporal feature
            channel_group_feature = temporal_feature

        # Select the features for the target variable(s)
        target_features = torch.stack(
            [
                channel_group_feature[i, self.target_positions]
                for i in range(channel_group_feature.size(0))
            ],
            dim=0,
        )

        # if torch.isnan(target_features).any():
        #     raise ValueError("Target features contain NaN values.")

        # Passing only the target features to the prediction head
        normalized_prediction = self.linear_head(target_features)

        # if torch.isnan(normalized_prediction).any():
        #     raise ValueError("Predictions contain NaN values.")

        normalized_prediction = rearrange(normalized_prediction, "b n d -> b d n")

        prediction = self.transform_output(
            prediction=normalized_prediction, target_scale=x["target_scale"]
        )

        return self.to_network_output(prediction=prediction, L_importance=L_importance)

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: list[str] = None,
        **kwargs,
    ):
        """
        Create a DUET model from a TimeSeriesDataSet.
        """
        new_kwargs = DEFAULT_DUET_HYPER_PARAMS.copy()
        new_kwargs.update(kwargs)

        # Adding parameters we infer from the dataset
        new_kwargs.update(
            {
                "seq_len": dataset.max_encoder_length,
                "pred_len": dataset.max_prediction_length,
                "enc_in": len(dataset.reals) + len(dataset.categoricals),
            }
        )

        return super().from_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            **new_kwargs,
        )
