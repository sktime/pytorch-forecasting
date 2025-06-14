"""
Time Series Transformer with eXogenous variables (TimeXer)
---------------------------------------------------------
"""

from copy import copy
from typing import Optional, Union
import warnings as warn

import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    MASE,
    RMSE,
    SMAPE,
    MultiHorizonMetric,
    QuantileLoss,
)
from pytorch_forecasting.metrics.base_metrics import MultiLoss
from pytorch_forecasting.models.base import BaseModelWithCovariates
from pytorch_forecasting.models.timexer.sub_modules import (
    AttentionLayer,
    DataEmbedding_inverted,
    Encoder,
    EncoderLayer,
    EnEmbedding,
    FlattenHead,
    FullAttention,
)


class TimeXer(BaseModelWithCovariates):
    """TimeXer model for time series forecasting with exogenous variables."""

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        task_name: str = "long_term_forecast",
        features: str = "MS",
        enc_in: int = None,
        hidden_size: int = 256,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.2,
        activation: str = "relu",
        patch_length: int = 16,
        factor: int = 5,
        embed_type: str = "fixed",
        freq: str = "h",
        output_size: Union[int, list[int]] = 1,
        loss: MultiHorizonMetric = None,
        learning_rate: float = 1e-3,
        static_categoricals: Optional[list[str]] = None,
        static_reals: Optional[list[str]] = None,
        time_varying_categoricals_encoder: Optional[list[str]] = None,
        time_varying_categoricals_decoder: Optional[list[str]] = None,
        time_varying_reals_encoder: Optional[list[str]] = None,
        time_varying_reals_decoder: Optional[list[str]] = None,
        x_reals: Optional[list[str]] = None,
        x_categoricals: Optional[list[str]] = None,
        embedding_sizes: Optional[dict[str, tuple[int, int]]] = None,
        embedding_labels: Optional[list[str]] = None,
        embedding_paddings: Optional[list[str]] = None,
        categorical_groups: Optional[dict[str, list[str]]] = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """An implementation of the TimeXer model.

        TimeXer empowers the canonical transformer with the ability to reconcile
        endogenous and exogenous information without any architectural modifications
        and achieves consistent state-of-the-art performance across twelve real-world
        forecasting benchmarks.

        TimeXer employs patch-level and variate-level representations respectively for
        endogenous and exogenous variables, with an endogenous global token as a bridge
        in-between. With this design, TimeXer can jointly capture intra-endogenous
        temporal dependencies and exogenous-to-endogenous correlations.

        TimeXer model for time series forecasting with exogenous variables.

        Parameters
        ----------
        context_length (int): Length of input sequence used for making predictions.
        prediction_length (int): Number of future time steps to predict.
        task_name (str, optional): Type of forecasting task, either
            'long_term_forecast' or 'short_term_forecast', which corresponds to
            forecasting scenarios implied by the task names.
        features (str, optional): Type of features used in the model ('MS' for
            multivariate forecating with single target, 'M' for multivariate
            forecasting with multiple targets and 'S' for univariate forecasting).
        enc_in (int, optional): Number of input variables for encoder.
        hidden_size (int, optional): Dimension of model embeddings and hidden
            representations.
        n_heads (int, optional): Number of attention heads in multi-head attention
            layers.
        e_layers (int, optional): Number of encoder layers with dual attention
            mechanism.
        d_ff (int, optional): Dimension of feedforward network in transformer layers
        dropout (float, optional): Dropout rate applied throughout the network for
            regularization.
        activation (str, optional): Activation function used in feedforward networks
            ('relu' or 'gelu').
        patch_length (int, optional): Length of each non-overlapping patch for
            endogenous variable tokenization.
        use_norm (bool, optional): Whether to apply normalization to input data.
            Do not change, as it a setting controlled by the pytorch-forecasting API
        factor: Scaling factor for attention scores.
        embed_type: Type of time feature embedding ('timeF' for time-based features)
        freq: Frequency of the time series data('h' for hourly,'d' for daily, etc.).
        static_categoricals (list[str]): names of static categorical variables
        static_reals (list[str]): names of static continuous variables
        time_varying_categoricals_encoder (list[str]): names of categorical
            variables for encoder
        time_varying_categoricals_decoder (list[str]): names of categorical
            variables for decoder
        time_varying_reals_encoder (list[str]): names of continuous variables for
            encoder
        time_varying_reals_decoder (list[str]): names of continuous variables for
            decoder
        x_reals (list[str]): order of continuous variables in tensor passed to
            forward function
        x_categoricals (list[str]): order of categorical variables in tensor passed
            to forward function
        embedding_sizes (dict[str, tuple[int, int]]): dictionary mapping categorical
            variables to tuple of integers where the first integer denotes the
            number of categorical classes and the second the embedding size
        embedding_labels (dict[str, list[str]]): dictionary mapping (string) indices
            to list of categorical labels
        embedding_paddings (list[str]): names of categorical variables for which
            label 0 is always mapped to an embedding vector filled with zeros
        categorical_groups (dict[str, list[str]]): dictionary of categorical
            variables that are grouped together and can also take multiple values
            simultaneously (e.g. holiday during octoberfest). They should be
            implemented as bag of embeddings.
        logging_metrics (nn.ModuleList[LightningMetric]): list of metrics that are
            logged during training. Defaults to
            nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()]).
        **kwargs: additional arguments to :py:class:`~BaseModel`.
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
            if features == "M":
                loss = MultiLoss([MAE()] * len(self.target_positions))
            else:
                loss = MAE()
        self.save_hyperparameters(ignore=["loss", "logging_metrics"])
        # loss is a standalone module and is stored separately.
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        if self.hparams.context_length < self.hparams.patch_length:
            raise ValueError(
                f"context_length ({context_length}) must be greater than or equal to"
                f" patch_length ({patch_length}). Model cannot create patches larger"
                " than the sequence length."
            )

        if self.hparams.context_length % self.hparams.patch_length != 0:
            warn.warn(
                f"In the input sequence, the context_length ({context_length}) is not a"
                f" multiple of the patch_length ({patch_length}). This may lead to some"
                "patches being ignored during training."
            )

        self.patch_num = max(
            1, int(self.hparams.context_length // self.hparams.patch_length)
        )
        self.n_target_vars = len(self.target_positions)

        self.enc_in = enc_in
        if enc_in is None:
            self.enc_in = len(self.reals)

        self.n_quantiles = None

        if isinstance(loss, QuantileLoss):
            self.n_quantiles = len(loss.quantiles)

        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads}) "
                f"for the multi-head attention mechanism to work properly."
            )

        self.en_embedding = EnEmbedding(
            self.n_target_vars,
            self.hparams.hidden_size,
            self.hparams.patch_length,
            self.hparams.dropout,
        )

        self.ex_embedding = DataEmbedding_inverted(
            self.hparams.context_length,
            self.hparams.hidden_size,
            self.hparams.embed_type,
            self.hparams.freq,
            self.hparams.dropout,
        )

        if e_layers <= 0:
            raise ValueError(f"e_layers ({e_layers}) must be positive.")
        elif e_layers > 12:
            warn.warn(
                f"e_layers ({e_layers}) is quite high. This might lead to overfitting "
                f"and high computational cost. Consider using 2-6 layers.",
                UserWarning,
            )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.hparams.factor,
                            attention_dropout=self.hparams.dropout,
                            output_attention=False,
                        ),
                        self.hparams.hidden_size,
                        self.hparams.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.hparams.factor,
                            attention_dropout=self.hparams.dropout,
                            output_attention=False,
                        ),
                        self.hparams.hidden_size,
                        self.hparams.n_heads,
                    ),
                    self.hparams.hidden_size,
                    self.hparams.d_ff,
                    dropout=self.hparams.dropout,
                    activation=self.hparams.activation,
                )
                for l in range(self.hparams.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.hparams.hidden_size),
        )
        self.head_nf = self.hparams.hidden_size * (self.patch_num + 1)
        self.head = FlattenHead(
            self.enc_in,
            self.head_nf,
            self.hparams.prediction_length,
            head_dropout=self.hparams.dropout,
            n_quantiles=self.n_quantiles,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: list[str] = None,
        **kwargs,
    ):
        """
        Create model from dataset and set parameters related to covariates.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: list of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            TimeXer
        """  # noqa: E501
        new_kwargs = copy(kwargs)
        new_kwargs.update(
            {
                "context_length": dataset.max_encoder_length,
                "prediction_length": dataset.max_prediction_length,
            }
        )

        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, MAE()))

        return super().from_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            **new_kwargs,
        )

    def _forecast(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forecast for univariate or multivariate with single target (MS) case.

        Args:
            x: Dictionary containing entries for encoder_cat, encoder_cont
        """
        encoder_cont = x["encoder_cont"]
        encoder_time_idx = x.get("encoder_time_idx", None)
        target_pos = self.target_positions

        # masking to ignore the target variable
        mask = torch.ones(encoder_cont.shape[-1], dtype=torch.bool)
        mask[target_pos] = False
        exog_data = encoder_cont[..., mask]

        en_embed, n_vars = self.en_embedding(
            encoder_cont[:, :, target_pos[-1]].unsqueeze(-1).permute(0, 2, 1)
        )
        ex_embed = self.ex_embedding(exog_data, encoder_time_idx)

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
        Forecast for multivariate with multiple targets (M) case.

        Args:
            x: Dictionary containing entries for encoder_cat, encoder_cont
        Returns:
            Dictionary with predictions
        """

        encoder_cont = x["encoder_cont"]
        encoder_time_idx = x.get("encoder_time_idx", None)
        target_pos = self.target_positions
        encoder_target = encoder_cont[..., target_pos]

        en_embed, n_vars = self.en_embedding(encoder_target.permute(0, 2, 1))

        # use masking to ignore the target variable in encoder_cont under ex_embed.
        mask = torch.ones(
            encoder_cont.shape[-1], dtype=torch.bool, device=encoder_cont.device
        )
        mask[target_pos] = False
        exog_data = encoder_cont[..., mask]
        ex_embed = self.ex_embedding(exog_data, encoder_time_idx)

        # batch_size x sequence_length x hidden_size
        enc_out = self.encoder(en_embed, ex_embed)

        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )  # batch_size x n_vars x sequence_length x hidden_size

        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        if self.n_quantiles is not None:
            dec_out = dec_out.permute(0, 2, 1, 3)
        else:
            dec_out = dec_out.permute(0, 2, 1)

        return dec_out

    @property
    def decoder_covariate_size(self) -> int:
        """Decoder covariates size.

        Returns:
            int: size of time-dependent covariates used by the decoder
        """
        return len(
            set(self.hparams.time_varying_reals_decoder) - set(self.target_names)
        ) + sum(
            self.embeddings.output_size[name]
            for name in self.hparams.time_varying_categoricals_decoder
        )

    @property
    def encoder_covariate_size(self) -> int:
        """Encoder covariate size.

        Returns:
            int: size of time-dependent covariates used by the encoder
        """
        return len(
            set(self.hparams.time_varying_reals_encoder) - set(self.target_names)
        ) + sum(
            self.embeddings.output_size[name]
            for name in self.hparams.time_varying_categoricals_encoder
        )

    @property
    def static_size(self) -> int:
        """Static covariate size.

        Returns:
            int: size of static covariates
        """
        return len(self.hparams.static_reals) + sum(
            self.embeddings.output_size[name]
            for name in self.hparams.static_categoricals
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x: Dictionary containing model inputs

        Returns:
            Dictionary with model outputs
        """
        if (
            self.hparams.task_name == "long_term_forecast"
            or self.hparams.task_name == "short_term_forecast"
        ):  # noqa: E501
            if self.hparams.features == "M":
                out = self._forecast_multi(x)
            else:
                out = self._forecast(x)
            prediction = out[:, : self.hparams.prediction_length, :]

            target_positions = self.target_positions

            # note: prediction.size(2) is the number of target variables i.e n_targets
            target_indices = range(prediction.size(2))

            if prediction.size(2) != len(target_positions):
                prediction = prediction[:, :, : len(target_positions)]

            # In the case of a single target, the result will be a torch.Tensor
            # with shape (batch_size, prediction_length)
            # In the case of multiple targets, the result will be a list of "n_targets"
            # tensors with shape (batch_size, prediction_length)
            # If quantile predictions are used, the result will have an additional
            # dimension for quantiles, resulting in a shape of
            # (batch_size, prediction_length, n_quantiles)
            if self.n_quantiles is not None:
                # quantile predictions.
                if len(target_indices) == 1:
                    prediction = prediction[..., 0, :]
                else:
                    prediction = [prediction[..., i, :] for i in target_indices]
            else:
                # point predictions.
                if len(target_indices) == 1:
                    prediction = prediction[..., 0]
                else:
                    prediction = [prediction[..., i] for i in target_indices]
            prediction = self.transform_output(
                prediction=prediction, target_scale=x["target_scale"]
            )
            return self.to_network_output(prediction=prediction)
        else:
            return None
