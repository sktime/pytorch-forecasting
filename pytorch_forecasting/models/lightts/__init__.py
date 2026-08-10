"""
LightTS model for time series forecasting.
"""

from pytorch_forecasting.models.lightts._lightts_pkg_v2 import LightTS_pkg_v2
from pytorch_forecasting.models.lightts._lightts_v2 import LightTS

__all__ = ["LightTS", "LightTS_pkg_v2"]
