"""
Decomposition-Linear model for time series forecasting.
"""

from ptf.models.dlinear._dlinear_pkg_v2 import DLinear_pkg_v2
from ptf.models.dlinear._dlinear_v2 import DLinear

__all__ = ["DLinear", "DLinear_pkg_v2"]
