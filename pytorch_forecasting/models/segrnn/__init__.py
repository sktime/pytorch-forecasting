"""
SegRNN model for forecasting time series.
"""

from pytorch_forecasting.models.segrnn._segrnn_pkg_v2 import SegRNN_pkg_v2
from pytorch_forecasting.models.segrnn._segrnn_v2 import SegRNN

__all__ = [
    "SegRNN",
    "SegRNN_pkg_v2",
]
