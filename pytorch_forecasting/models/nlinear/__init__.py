"""
Normalization-Linear model for time series forecasting.
"""

from pytorch_forecasting.models.nlinear._nlinear_pkg_v2 import NLinear_pkg_v2
from pytorch_forecasting.models.nlinear._nlinear_v2 import NLinear

__all__ = ["NLinear", "NLinear_pkg_v2"]
