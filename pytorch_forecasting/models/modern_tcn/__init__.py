"""
Decomposition-Linear model for time series forecasting.
"""

from pytorch_forecasting.models.modern_tcn._modern_tcn_pkg_v2 import ModernTcn_pkg_v2
from pytorch_forecasting.models.modern_tcn._modern_tcn_v2 import ModernTcn

__all__ = ["ModernTcn", "ModernTcn_pkg_v2"]
