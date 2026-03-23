"""
UniTS layer abstractions - re-exported from canonical locations.
"""

from pytorch_forecasting.layers._blocks._transformer_block import _TransformerBlock
from pytorch_forecasting.layers._embeddings._patch_embedding import _PatchEmbedding
from pytorch_forecasting.layers._embeddings._positional_embedding import (
    _PositionalEmbedding as _PositionalEncoding,  # backward compat alias
)

__all__ = ["_PatchEmbedding", "_PositionalEncoding", "_TransformerBlock"]
