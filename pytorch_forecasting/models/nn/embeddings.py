"""Backward-compatible re-export of embedding layers.

The canonical location is ``pytorch_forecasting.layers._embeddings``.
"""

from pytorch_forecasting.layers._embeddings._multi_embedding import (
    MultiEmbedding,
    TimeDistributedEmbeddingBag,
)

__all__ = ["MultiEmbedding", "TimeDistributedEmbeddingBag"]
