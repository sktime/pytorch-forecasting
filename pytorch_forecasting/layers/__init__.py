"""
Architectural deep learning layers from `nn.Module`.
"""

from pytorch_forecasting.layers._attention import (
    AttentionLayer,
    FullAttention,
    TriangularCausalMask,
)
from pytorch_forecasting.layers._blocks import FreTSCore, ResidualBlock
from pytorch_forecasting.layers._decomposition import SeriesDecomposition
from pytorch_forecasting.layers._embeddings import (
    DataEmbedding_inverted,
    EnEmbedding,
    PatchEmbedding,
    PositionalEmbedding,
    embedding_cat_variables,
)
from pytorch_forecasting.layers._encoders import (
    Encoder,
    EncoderLayer,
)
from pytorch_forecasting.layers._mlp import FullyConnectedModule
from pytorch_forecasting.layers._normalization import RevIN
from pytorch_forecasting.layers._output._flatten_head import (
    FlattenHead,
)
from pytorch_forecasting.layers._output._patch_tst_flatten_head import (
    PatchTSTFlattenHead,
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
    "FullAttention",
    "AttentionLayer",
    "TriangularCausalMask",
    "DataEmbedding_inverted",
    "EnEmbedding",
    "PositionalEmbedding",
    "PatchEmbedding",
    "Encoder",
    "EncoderLayer",
    "FlattenHead",
    "PatchTSTFlattenHead",
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
    "FreTSCore",
    "FullyConnectedModule",
]
