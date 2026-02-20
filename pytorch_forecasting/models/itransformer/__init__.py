"""
iTransformer model for forecasting time series.
"""

from pytorch_forecasting.models.itransformer._itransformer_pkg_v2 import (
    iTransformer_pkg_v2,
)
from pytorch_forecasting.models.itransformer._itransformer_v2 import iTransformer

__all__ = [
    "iTransformer",
    "iTransformer_pkg_v2",
]
