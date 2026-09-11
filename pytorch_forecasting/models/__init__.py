"""
Models for timeseries forecasting.
"""

from pytorch_forecasting.models.base import (
    AutoRegressiveBaseModel,
    AutoRegressiveBaseModelWithCovariates,
    BaseModel,
    BaseModelWithCovariates,
)
from pytorch_forecasting.models.baseline import Baseline
from pytorch_forecasting.models.deepar import DeepAR
from pytorch_forecasting.models.frets import FreTS, FreTS_pkg_v2
from pytorch_forecasting.models.mlp import DecoderMLP
from pytorch_forecasting.models.nbeats import NBeats, NBeatsKAN
from pytorch_forecasting.models.nhits import NHiTS
from pytorch_forecasting.models.nn import GRU, LSTM, MultiEmbedding, get_rnn
from pytorch_forecasting.models.patch_tst import (
    PatchTST,
    PatchTST_pkg,
    PatchTST_pkg_v2,
    PatchTST_v2,
)
from pytorch_forecasting.models.rnn import RecurrentNetwork
from pytorch_forecasting.models.scinet import SCINet_pkg_v2, SCINet_v2
from pytorch_forecasting.models.softs import SOFTS, SOFTS_pkg_v2
from pytorch_forecasting.models.temporal_fusion_transformer import (
    TemporalFusionTransformer,
)
from pytorch_forecasting.models.tide import TiDEModel
from pytorch_forecasting.models.timexer import TimeXer
from pytorch_forecasting.models.xlstm import xLSTMTime

__all__ = [
    "NBeats",
    "NBeatsKAN",
    "NHiTS",
    "PatchTST",
    "PatchTST_v2",
    "PatchTST_pkg",
    "PatchTST_pkg_v2",
    "TemporalFusionTransformer",
    "RecurrentNetwork",
    "DeepAR",
    "BaseModel",
    "Baseline",
    "BaseModelWithCovariates",
    "AutoRegressiveBaseModel",
    "AutoRegressiveBaseModelWithCovariates",
    "get_rnn",
    "LSTM",
    "GRU",
    "MultiEmbedding",
    "DecoderMLP",
    "TiDEModel",
    "TimeXer",
    "xLSTMTime",
    "SOFTS",
    "SOFTS_pkg_v2",
    "SCINet_v2",
    "SCINet_pkg_v2",
    "FreTS",
    "FreTS_pkg_v2",
]
