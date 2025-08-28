"""
Decomposition-Linear model for time series forecasting.
"""

from pytorch_forecasting.models.dlinear._dlinear_pkg_v2 import DLinear_pkg_v2
from pytorch_forecasting.models.dlinear._dlinear_v2 import DLinear

__all__ = ["DLinear", "DLinear_pkg_v2"]
