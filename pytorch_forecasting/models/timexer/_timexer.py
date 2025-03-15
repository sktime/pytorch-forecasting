"""
Time Series Transformer with eXogenous variables (TimeXer)
---------------------------------------------------------
"""

import copy
from typing import Dict, List, Optional, Tuple

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
    EnEmbeddding,
    FlattenHead,
    FullAttention,
    PositionalEmbedding,
)


class TimeXer(BaseModelWithCovariates):
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        task_name: str = "long_term_forecast",
        features: str = "MS",
        enc_in: int = None,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        patch_length: int = 24,
        use_norm: bool = True,
        factor: int = 5,
        embed_type: str = "fixed",
        freq: str = "h",
        loss: MultiHorizonMetric = None,
        learning_rate: float = 1e-3,
        static_categoricals: Optional[List[str]] = None,
        static_reals: Optional[List[str]] = None,
        time_varying_categoricals_encoder: Optional[List[str]] = None,
        time_varying_categoricals_decoder: Optional[List[str]] = None,
        time_varying_reals_encoder: Optional[List[str]] = None,
        time_varying_reals_decoder: Optional[List[str]] = None,
        x_reals: Optional[List[str]] = None,
        x_categoricals: Optional[List[str]] = None,
        embedding_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
        embedding_labels: Optional[List[str]] = None,
        embedding_paddings: Optional[List[str]] = None,
        categorical_groups: Optional[Dict[str, List[str]]] = None,
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

        Args:

            context_length (int): Length of input sequence used for making predictions.
            prediction_length (int): Number of future time steps to predict.
            task_name (str, optional): Type of forecasting task, either
                'long_term_forecast' or 'short_term_forecast', which corresponds to
                forecasting scenarios implied by the task names.
            features (str, optional): Type of features used in the model ('MS' for
                multivariate forecating with single target, 'M' for multivariate
                forecasting with multiple targets and 'S' for univariate forecasting).
            enc_in (int, optional): Number of input variables for encoder.
            d_model (int, optional): Dimension of model embeddings and hidden
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
            factor: Scaling factor for attention scores.
            embed_type: Type of time feature embedding ('timeF' for time-based features)
            freq: Frequency of the time series data('h' for hourly,'d' for daily, etc.).
            static_categoricals (List[str]): names of static categorical variables
            static_reals (List[str]): names of static continuous variables
            time_varying_categoricals_encoder (List[str]): names of categorical
                variables for encoder
            time_varying_categoricals_decoder (List[str]): names of categorical
                variables for decoder
            time_varying_reals_encoder (List[str]): names of continuous variables for
                encoder
            time_varying_reals_decoder (List[str]): names of continuous variables for
                decoder
            x_reals (List[str]): order of continuous variables in tensor passed to
                forward function
            x_categoricals (List[str]): order of categorical variables in tensor passed
                to forward function
            embedding_sizes (Dict[str, Tuple[int, int]]): dictionary mapping categorical
                variables to tuple of integers where the first integer denotes the
                number of categorical classes and the second the embedding size
            embedding_labels (Dict[str, List[str]]): dictionary mapping (string) indices
                to list of categorical labels
            embedding_paddings (List[str]): names of categorical variables for which
                label 0 is always mapped to an embedding vector filled with zeros
            categorical_groups (Dict[str, List[str]]): dictionary of categorical
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
            loss = QuantileLoss()

        self.save_hyperparameters()
        # loss is a standalone module and is stored separately.
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        # [x] todo: implement the from_dataset method
        # [] todo: implement the forward and forecast methods into the class.
        self.patch_num = max(1, int(context_length // patch_length))
        self.n_target_vars = len(self.target_positions)

        if enc_in is None:
            if features == "MS":
                self.enc_in = 1
            else:
                self.enc_in = self.n_target_vars
        else:
            self.enc_in = enc_in

        self.n_vars = 1 if self.features == "MS" else self.enc_in

        self.en_embedding = EnEmbeddding(
            self.hparams.n_vars,
            self.hparams.d_model,
            self.hparams.patch_length,
            self.hparams.dropout,
        )

        self.ex_embedding = DataEmbedding_inverted(
            self.hparams.context_length,
            self.hparams.d_model,
            self.hparams.embed_type,
            self.hparams.freq,
            self.hparams.dropout,
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
                        self.hparams.d_model,
                        self.hparams.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            True,
                            self.hparams.factor,
                            attention_dropout=self.hparams.dropout,
                            output_attention=False,
                        ),
                        self.hparams.d_model,
                        self.hparams.n_heads,
                    ),
                    self.hparams.d_model,
                    self.hparams.d_ff,
                    dropout=self.hparams.dropout,
                    activation=self.hparams.activation,
                )
                for l in range(self.hparams.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.hparams.d_model),
        )
        self.head_nf = self.hparams.d_model * (
            self.params.context_length // self.hparams.patch_length
        )
        self.head = FlattenHead(
            self.enc_in, self.hparms.head_nf, self.hparams.prediction_length
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: List[str] = None,
        **kwargs,
    ) -> LightningModule:
        """
        Create model from dataset and set parameters related to covariates.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
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

        new_kwargs.update(
            cls.deduce_default_output_parameters(
                dataset, kwargs, MultiLoss[SMAPE(), MAE()]
            )
        )  # noqa: E501

        return super().from_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            **new_kwargs,
        )
