"""
Autoformer model for time series forecasting.
"""

from pytorch_forecasting.models.autoformer._autoformer_pkg_v2 import Autoformer_pkg_v2
from pytorch_forecasting.models.autoformer._autoformer_v2 import Autoformer

__all__ = ["Autoformer", "Autoformer_pkg_v2"]
