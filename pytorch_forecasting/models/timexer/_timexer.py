"""
Time Series Transformer with eXogenous variables (TimeXer)
---------------------------------------------------------
"""

from typing import Dict, List, Optional, Tuple

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
from pytorch_forecasting.models.base import BaseModelWithCovariates
from pytorch_forecasting.models.timexer.sub_modules import (
    AttentionLayer,
    DataEmbedding_inverted,
    FullAttention,
    PositionalEmbedding,
)


class TimeXer(BaseModelWithCovariates):
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        task_name: str = "long_term_forecast",
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

        # todo: implement the model from https://github.com/thuml/TimeXer
