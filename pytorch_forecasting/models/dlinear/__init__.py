"""
Decomposition-Linear model for time series forecasting.
"""

from pytorch_forecasting.models.dlinear._dlinear_pkg_v2 import DLinearModel_pkg_v2
from pytorch_forecasting.models.dlinear._dlinear_v2 import DLinearModel

__all__ = [
    "DLinearModel" "DLinearModel_pkg_v2",
]
