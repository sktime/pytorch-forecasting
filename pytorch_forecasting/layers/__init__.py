"""
Architectural deep learning layers from `nn.Module`.
"""

from pytorch_forecasting.layers._attention import (
    AttentionLayer,
    AutoCorrelation,
    AutoCorrelationLayer,
    FullAttention,
    TriangularCausalMask,
)
from pytorch_forecasting.layers._blocks import ResidualBlock
from pytorch_forecasting.layers._decoders import (
    AutoformerDecoder,
    AutoformerDecoderLayer,
)
from pytorch_forecasting.layers._decomposition import SeriesDecomposition
from pytorch_forecasting.layers._embeddings import (
    DataEmbedding_inverted,
    DataEmbedding_wo_pos,
    EnEmbedding,
    FixedEmbedding,
    PositionalEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
    TokenEmbedding,
    embedding_cat_variables,
)
from pytorch_forecasting.layers._encoders import (
    AutoformerEncoder,
    AutoformerEncoderLayer,
    Encoder,
    EncoderLayer,
)
from pytorch_forecasting.layers._mlp import FullyConnectedModule
from pytorch_forecasting.layers._normalization import RevIN, SeasonalLayerNorm
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

__all__ = [
    "AttentionLayer",
    "AutoCorrelation",
    "AutoCorrelationLayer",
    "AutoformerDecoder",
    "AutoformerDecoderLayer",
    "AutoformerEncoder",
    "AutoformerEncoderLayer",
    "DataEmbedding_inverted",
    "DataEmbedding_wo_pos",
    "EnEmbedding",
    "Encoder",
    "EncoderLayer",
    "FixedEmbedding",
    "FlattenHead",
    "FullAttention",
    "FullyConnectedModule",
    "PositionalEmbedding",
    "ResidualBlock",
    "RevIN",
    "SeasonalLayerNorm",
    "SeriesDecomposition",
    "TemporalEmbedding",
    "TimeFeatureEmbedding",
    "TokenEmbedding",
    "TriangularCausalMask",
    "embedding_cat_variables",
    "mLSTMCell",
    "mLSTMLayer",
    "mLSTMNetwork",
    "sLSTMCell",
    "sLSTMLayer",
    "sLSTMNetwork",
]
