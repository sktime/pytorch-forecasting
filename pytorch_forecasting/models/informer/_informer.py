"""
Informer Transformer for Long Sequence Time-Series Forecasting.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric
from pytorch_forecasting.models.base import BaseModel
from pytorch_forecasting.utils._dependencies import _check_matplotlib


class Informer(BaseModel):
    def __init__(
        self,
        encoder_input: int,
        decoder_input: int,
        out_channels: int,
        seq_len: int,
        label_len: int,
        out_len: int,
        factor: int = 5,
        d_model: int = 512,
        n_heads: int = 8,
        encoder_layers: Union[int, List[int]] = 3,
        decoder_layers: int = 2,
        d_ff: int = 512,
        dropout: int = 0.0,
        attn: str = "prob",
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        distil: bool = True,
        mix: bool = True,
        logging_metrics: Optional[nn.ModuleList] = None,
        **kwargs,
    ):
        super().__init__()
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
