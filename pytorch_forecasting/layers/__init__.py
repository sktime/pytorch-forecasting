"""
Architectural deep learning layers from `nn.Module`.
"""

from pytorch_forecasting.layers._attention import (
    AttentionLayer,
    FullAttention,
    TriangularCausalMask,
)
from pytorch_forecasting.layers._blocks import ResidualBlock
from pytorch_forecasting.layers._decomposition import SeriesDecomposition
from pytorch_forecasting.layers._embeddings import (
    DataEmbedding,
    DataEmbedding_inverted,
    EnEmbedding,
    FixedEmbedding,
    PositionalEmbedding,
    TemporalEmbedding,
    TokenEmbedding,
    embedding_cat_variables,
)
from pytorch_forecasting.layers._encoders import (
    Encoder,
    EncoderLayer,
)
from pytorch_forecasting.layers._normalization import RevIN
from pytorch_forecasting.layers._output._flatten_head import (
    FlattenHead,
)
from pytorch_forecasting.layers._recurrent._mlstm import (
    mLSTMCell,
    mLSTMLayer,
    mLSTMNetwork,
)
from pytorch_forecasting.layers._recurrent._slstm import (
    sLSTMCell,
    sLSTMLayer,
    sLSTMNetwork,
)
from pytorch_forecasting.layers._reformer import (
    ReformerEncoder,
    ReformerEncoderLayer,
    ReformerLayer,
)

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
    "RevIN",
    "ResidualBlock",
    "embedding_cat_variables",
    "ReformerEncoder",
    "ReformerEncoderLayer",
    "ReformerLayer",
    "DataEmbedding",
    "TemporalEmbedding",
    "FixedEmbedding",
    "TokenEmbedding",
]
