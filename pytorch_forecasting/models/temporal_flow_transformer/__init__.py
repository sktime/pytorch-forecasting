"""
Implementation of the Temporal Flow Transformer.

This network is capable of modelling the dependencies of thousands of time series amongst each other.
The model is also based on the "Transfomer-MAF" in
`Multi-variate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows
<https://arxiv.org/abs/2002.06103>`_.
"""
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from pytorch_forecasting.metrics import FlowDistributionLoss
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.temporal_flow_transformer.submodules import MAF, RealNVP


class TemporalFlowTransformer(BaseModelWithCovariates):
    def __init__(
        self,
        # network arguments
        dequantize: bool,
        hidden_transformer_size: int = 5,
        hidden_flow_size: int = 5,
        n_flow_blocks: int = 2,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        n_hidden_flow_layers: int = 3,
        history_length: int = 10,
        context_length: int = 2,
        conditioning_length: int = 4,
        prediction_length: int = 5,
        attention_head_size: int = 2,
        transformer_dim_feedforward_multiplier: int = 2,
        dropout: float = 0.1,
        act_type: str = "ReLU",
        flow_type: str = "MAF",
        # standard arguments
        loss: FlowDistributionLoss = None,
        logging_metrics: nn.ModuleList = None,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        hidden_continuous_size: int = 8,
        hidden_continuous_sizes: Dict[str, int] = {},
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        learning_rate: float = 1e-3,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        log_gradient_flow: bool = False,
        reduce_on_plateau_patience: int = 1000,
        monotone_constaints: Dict[str, int] = {},
        **kwargs
    ) -> None:
        if loss is None:
            loss = FlowDistributionLoss()

        self.save_hyperparameters()
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        self.encoder_input = nn.Linear(self.input_size, hidden_transformer_size)
        self.decoder_input = nn.Linear(self.input_size, hidden_transformer_size)

        # [B, T, d_model] where d_model / num_heads is int
        self.transformer = nn.Transformer(
            d_model=hidden_transformer_size,
            nhead=attention_head_size,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=transformer_dim_feedforward_multiplier * hidden_transformer_size,
            dropout=dropout,
            activation=act_type,
        )

        flow_cls = {
            "RealNVP": RealNVP,
            "MAF": MAF,
        }[flow_type]
        self.flow = flow_cls(
            input_size=self.target_dim,
            n_blocks=n_flow_blocks,
            n_hidden=n_hidden_flow_layers,
            hidden_size=hidden_flow_size,
            cond_label_size=conditioning_length,
        )

        self.embed_dim = 1
        self.embed = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.embed_dim)

        # mask
        self.register_buffer(
            "tgt_mask",
            self.transformer.generate_square_subsequent_mask(prediction_length),
        )
