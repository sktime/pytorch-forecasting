"""
TimeMoE model for forecasting time series.
"""

from pytorch_forecasting.models.timemoe._timemoe_pkg_v2 import TimeMoE_pkg_v2
from pytorch_forecasting.models.timemoe._timemoe_v2 import TimeMoE

__all__ = [
    "TimeMoE",
    "TimeMoE_pkg_v2",
]
