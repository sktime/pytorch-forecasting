"""
PatchTST model for time series forecasting.
"""

from copy import copy
from typing import Optional, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.layers import (
    PatchEmbedding,
    PatchTSTFlattenHead as FlattenHead,
)
from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE, MultiLoss, QuantileLoss
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric
from pytorch_forecasting.models.base import BaseModelWithCovariates
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding


class PatchTST(BaseModelWithCovariates):
    """PatchTST model for time series forecasting."""

    @classmethod
    def _pkg(cls):
        """Package for the model."""
        from pytorch_forecasting.models.patch_tst._patch_tst_pkg import PatchTST_pkg

        return PatchTST_pkg

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: int = 0,
        d_model: int = 128,
        n_heads: int = 16,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        fc_dropout: float = 0.2,
        head_dropout: float = 0.0,
        activation: str = "gelu",
        output_size: int | list[int] = 1,
        loss: MultiHorizonMetric = None,
        learning_rate: float = 1e-3,
        static_categoricals: list[str] | None = None,
        static_reals: list[str] | None = None,
        time_varying_categoricals_encoder: list[str] | None = None,
        time_varying_categoricals_decoder: list[str] | None = None,
        categorical_groups: dict[str, list[str]] | None = None,
        time_varying_reals_encoder: list[str] | None = None,
        time_varying_reals_decoder: list[str] | None = None,
        embedding_sizes: dict[str, tuple[int, int]] | None = None,
        embedding_paddings: list[str] | None = None,
        embedding_labels: dict[str, list[str]] | None = None,
        x_reals: list[str] | None = None,
        x_categoricals: list[str] | None = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """
        Implementation of the PatchTST model.

        PatchTST leverages Channel Independence and patching of time series to
        achieve state-of-the-art performance on long-term forecasting.

        Parameters
        ----------
        context_length : int
            Length of input sequence used for making predictions.
        prediction_length : int
            Number of future time steps to predict.
        patch_len : int
            Length of each patch.
        stride : int
            Stride of the patch.
        padding_patch : int
            Padding at the end of the sequence for patching.
        d_model : int
            Dimension of the model.
        n_heads : int
            Number of attention heads.
        e_layers : int
            Number of encoder layers.
        d_ff : int
            Dimension of the feed-forward network.
        dropout : float
            Dropout rate.
        fc_dropout : float
            Dropout rate for fully connected layers.
        head_dropout : float
            Dropout rate for the flatten head.
        activation : str
            Activation function ('relu' or 'gelu').
        """
        if static_categoricals is None:
            static_categoricals = []
        if static_reals is None:
            static_reals = []
        if time_varying_categoricals_encoder is None:
            time_varying_categoricals_encoder = []
        if time_varying_categoricals_decoder is None:
            time_varying_categoricals_decoder = []
        if categorical_groups is None:
            categorical_groups = {}
        if time_varying_reals_encoder is None:
            time_varying_reals_encoder = []
        if time_varying_reals_decoder is None:
            time_varying_reals_decoder = []
        if embedding_sizes is None:
            embedding_sizes = {}
        if embedding_paddings is None:
            embedding_paddings = []
        if embedding_labels is None:
            embedding_labels = {}
        if x_reals is None:
            x_reals = []
        if x_categoricals is None:
            x_categoricals = []
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])

        if loss is None:
            loss = MAE()

        self.save_hyperparameters(ignore=["loss", "logging_metrics"])
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        self.output_dim = len(self.target_names)

        self.embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
        )

        # compute number of patches
        rem = self.hparams.context_length % self.hparams.patch_len
        self.pad_len = 0 if rem == 0 else self.hparams.patch_len - rem

        self.patch_num = (
            int(
                (
                    self.hparams.context_length
                    + self.pad_len
                    + self.hparams.padding_patch
                    - self.hparams.patch_len
                )
                / self.hparams.stride
            )
            + 1
        )

        self.n_quantiles = 1
        if isinstance(loss, QuantileLoss):
            self.n_quantiles = len(loss.quantiles)
        elif isinstance(loss, MultiLoss):
            # For multi-target, check if any is QuantileLoss
            if any(isinstance(l, QuantileLoss) for l in loss.losses):
                # Assume all are QuantileLoss with same quantiles if one is
                for l in loss.losses:
                    if isinstance(l, QuantileLoss):
                        self.n_quantiles = len(l.quantiles)
                        break

        # Architecture components
        self.patch_embedding = PatchEmbedding(
            self.hparams.d_model,
            self.hparams.patch_len,
            self.hparams.stride,
            self.hparams.padding_patch,
            self.hparams.dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.d_model,
            nhead=self.hparams.n_heads,
            dim_feedforward=self.hparams.d_ff,
            dropout=self.hparams.dropout,
            activation=self.hparams.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.hparams.e_layers
        )

        self.flatten_head = FlattenHead(
            self.patch_num,
            self.hparams.d_model,
            self.hparams.prediction_length,
            head_dropout=self.hparams.head_dropout,
            n_quantiles=self.n_quantiles,
        )

    @property
    def decoder_covariate_size(self) -> int:
        return len(
            set(self.hparams.time_varying_reals_decoder) - set(self.target_names)
        ) + sum(
            self.embeddings.output_size[name]
            for name in self.hparams.time_varying_categoricals_decoder
        )

    @property
    def encoder_covariate_size(self) -> int:
        return len(
            set(self.hparams.time_varying_reals_encoder) - set(self.target_names)
        ) + sum(
            self.embeddings.output_size[name]
            for name in self.hparams.time_varying_categoricals_encoder
        )

    @property
    def static_size(self) -> int:
        return len(self.hparams.static_reals) + sum(
            self.embeddings.output_size[name]
            for name in self.hparams.static_categoricals
        )

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        new_kwargs = copy(kwargs)
        new_kwargs.update(
            {
                "context_length": dataset.max_encoder_length,
                "prediction_length": dataset.max_prediction_length,
            }
        )
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, MAE()))
        return super().from_dataset(dataset, **new_kwargs)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # target
        encoder_y = x["encoder_cont"][..., self.target_positions]

        # covariates
        encoder_features = self.extract_features(x, self.embeddings, period="encoder")

        if self.encoder_covariate_size > 0 or self.static_size > 0:
            covariates = []
            if self.static_size > 0:
                # Add static covariates expanded over the time dimension
                for name in self.static_variables:
                    cov = encoder_features[name]
                    if cov.ndim == 2:
                        cov = cov.unsqueeze(1).expand(-1, encoder_y.shape[1], -1)
                    covariates.append(cov)

            if self.encoder_covariate_size > 0:
                for name in self.encoder_variables:
                    if name not in self.target_names:
                        covariates.append(encoder_features[name])

            if len(covariates) > 0:
                encoder_x_t = torch.cat(covariates, dim=2)
                input_vector = torch.cat((encoder_y, encoder_x_t), dim=2)
            else:
                input_vector = encoder_y
        else:
            input_vector = encoder_y

        batch_size, seq_len, total_vars = input_vector.shape

        # PatchTST Channel Independence: treat total_vars as independent batches
        # [Batch, seq_len, total_vars] -> [Batch, total_vars, seq_len]
        # -> [Batch * total_vars, 1, seq_len]
        enc_in = input_vector.permute(0, 2, 1).reshape(
            batch_size * total_vars, 1, seq_len
        )

        # Pad sequence if necessary to match the expected patch_num
        # This handles cases where seq_len is not perfectly divisible by patch_len
        # or if a batch sequence is shorter than context_length.
        total_pad_len = max(0, self.hparams.context_length + self.pad_len - seq_len)
        if total_pad_len > 0:
            enc_in = F.pad(enc_in, (0, total_pad_len))

        # [Batch * total_vars, PatchNum, d_model]
        enc_out = self.patch_embedding(enc_in)

        # [Batch * total_vars, PatchNum, d_model]
        enc_out = self.encoder(enc_out)

        # [Batch * total_vars, PredLen, n_quantiles]
        dec_out = self.flatten_head(enc_out)

        # Reshape back to separate Batch and Variables
        # [Batch, total_vars, PredLen, n_quantiles]
        dec_out = dec_out.reshape(
            batch_size, total_vars, self.hparams.prediction_length, self.n_quantiles
        )

        # Extract target predictions (they are the first output_dim variables)
        prediction = dec_out[:, : self.output_dim, :, :]

        # [Batch, PredLen, output_dim, n_quantiles]
        prediction = prediction.permute(0, 2, 1, 3)

        if self.output_dim == 1:
            prediction = prediction[..., 0, :]
        else:
            # output format for multi-targets is a list of tensors
            prediction = [prediction[..., i, :] for i in range(self.output_dim)]

        prediction = self.transform_output(prediction, target_scale=x["target_scale"])
        return self.to_network_output(prediction=prediction)
