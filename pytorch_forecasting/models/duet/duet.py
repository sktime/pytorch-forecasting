from copy import copy
from logging import config
from typing import Callable, Optional, Union

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric

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
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "stride": 8,
    "period_len": 4,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "moving_avg": 25,
    "batch_size": 256,
    "lradj": "type3",
    "lr": 0.02,
    "num_epochs": 1,  # <--- Set to 1 for quick testing, default 100
    "num_workers": 0,
    "loss": "huber",
    "patience": 10,
    "num_experts": 4,
    "noisy_gating": True,
    "k": 1,
    "CI": False,
    "parallel_strategy": "DP",
}


class DUETModel(BaseModelWithCovariates):
    def __init__(
        self,
        # --- Parameters for BaseModelWithCovariates (explicitly listed) ---
        # These are "consumed" here and not passed to the parent's __init__
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
        # --- Parameters for DUET's architecture ---
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
        fc_dropout: float,
        activation: str,
        output_attention: bool,
        factor: int,
        CI: bool,
        num_experts: int,
        noisy_gating: bool,
        k: int,
        # --- Parameters for the BaseModel parent (will be passed up) ---
        loss: Metric,
        learning_rate: float,
        dataset_parameters: dict,
        optimizer: str = "adam",
        output_transformer: Callable = None,
        # Standard BaseModel parameters
        **kwargs,
    ):
        self.save_hyperparameters()
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            optimizer=optimizer,
            output_transformer=output_transformer,
            dataset_parameters=dataset_parameters,
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
        input_tensor = x["encoder_cont"]
        batch_size = input_tensor.shape[0]

        if torch.isnan(input_tensor).any():
            raise ValueError("Input tensor contains NaN values.")

        if self.hparams.CI:
            channel_independent_input = rearrange(input_tensor, "b l n -> (b n) l 1")

            if torch.isnan(channel_independent_input).any():
                raise ValueError("Input tensor contains NaN values.")

            reshaped_output, L_importance = self.cluster(channel_independent_input)

            if torch.isnan(reshaped_output).any():
                raise ValueError("Input tensor contains NaN values.")

            temporal_feature = rearrange(
                reshaped_output, "(b n) d -> b d n", b=batch_size
            )  # noqa: E501
        else:
            temporal_feature, L_importance = self.cluster(input_tensor)

        if torch.isnan(temporal_feature).any():
            raise ValueError("Input tensor contains NaN values.")

        temporal_feature = rearrange(temporal_feature, "b d n -> b n d")

        if torch.isnan(temporal_feature).any():
            raise ValueError("Input tensor contains NaN values #2.")

        if len(self.hparams.time_varying_reals_encoder) > 1:
            # Multivariate case: apply channel transformer

            changed_input = rearrange(input_tensor, "b l n -> b n l")

            if torch.isnan(changed_input).any():
                raise ValueError("Input tensor contains NaN values.")

            channel_mask = self.mask_generator(changed_input)

            if torch.isnan(channel_mask).any():
                raise ValueError("Input tensor contains NaN values.")

            channel_group_feature, _ = self.Channel_transformer(
                x=temporal_feature, attn_mask=channel_mask
            )

            if torch.isnan(channel_group_feature).any():
                raise ValueError("Input tensor contains NaN values.")
        else:
            # For univariate case, the group feature is just the temporal feature
            channel_group_feature = temporal_feature

        # <<<<<<<<<<<<<<<< START OF FIX >>>>>>>>>>>>>>>>
        # We have features for all channels in `channel_group_feature`
        # (shape: batch_size, n_channels, d_model)
        # We only want to predict the target(s). `self.target_positions`
        # gives us their indices.

        # Select the features for the target variable(s)
        target_features = torch.stack(
            [
                channel_group_feature[i, self.target_positions]
                for i in range(channel_group_feature.size(0))
            ],
            dim=0,
        )

        if torch.isnan(target_features).any():
            raise ValueError("Target features contain NaN values.")

        # Pass only the target features to the prediction head
        normalized_prediction = self.linear_head(target_features)

        if torch.isnan(normalized_prediction).any():
            raise ValueError("Predictions contain NaN values.")

        normalized_prediction = rearrange(normalized_prediction, "b n p -> b p n")

        if torch.isnan(normalized_prediction).any():
            raise ValueError("Predictions contain NaN value.")

        prediction = self.transform_output(
            prediction=normalized_prediction, target_scale=x["target_scale"]
        )

        if torch.isnan(prediction).any():
            raise ValueError("Predictions contain NaN values.")

        return self.to_network_output(prediction=prediction, L_importance=L_importance)

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: list[
            str
        ] = None,  # Match parent signature
        **kwargs,
    ):
        """
        Create a DUET model from a TimeSeriesDataSet.
        """
        new_kwargs = DEFAULT_DUET_HYPER_PARAMS.copy()
        new_kwargs.update(kwargs)

        # Add parameters that we can infer from the dataset for DUET's specific needs
        new_kwargs.update(
            {
                "seq_len": dataset.max_encoder_length,
                "pred_len": dataset.max_prediction_length,
                "enc_in": len(dataset.reals) + len(dataset.categoricals),
            }
        )

        print("------------------------In DUETModel.from_dataset()")
        print(new_kwargs)

        return super().from_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            **new_kwargs,
        )
