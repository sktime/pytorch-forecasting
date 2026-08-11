"""Modern Temporal Convolutional Network model for time series forecasting."""

from pytorch_forecasting.models.modern_tcn._modern_tcn_pkg_v2 import ModernTCN_pkg_v2
from pytorch_forecasting.models.modern_tcn._modern_tcn_v2 import ModernTCN

__all__ = ["ModernTCN", "ModernTCN_pkg_v2"]
