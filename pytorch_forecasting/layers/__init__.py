"""
Architectural deep learning layers from `nn.Module`.
"""

from pytorch_forecasting.layers._attention import (
    AttentionLayer,
    FullAttention,
    TriangularCausalMask,
)
from pytorch_forecasting.layers._decomposition import SeriesDecomposition
from pytorch_forecasting.layers._embeddings import (
    DataEmbedding_inverted,
    EnEmbedding,
    PositionalEmbedding,
)
from pytorch_forecasting.layers._encoders import (
    Encoder,
    EncoderLayer,
)
from pytorch_forecasting.layers._mlstm import mLSTMCell, mLSTMLayer, mLSTMNetwork
from pytorch_forecasting.layers._output._flatten_head import (
    FlattenHead,
)
from pytorch_forecasting.layers._slstm import sLSTMCell, sLSTMLayer, sLSTMNetwork

__all__ = [
    "FullAttention",
    "AttentionLayer",
    "TriangularCausalMask",
    "DataEmbedding_inverted",
    "EnEmbedding",
    "PositionalEmbedding",
    "Encoder",
    "EncoderLayer",
    "FlattenHead",
    "mLSTMCell",
    "mLSTMLayer",
    "mLSTMNetwork",
    "sLSTMCell",
    "sLSTMLayer",
    "sLSTMNetwork",
    "SeriesDecomposition",
]
