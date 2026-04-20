"""
PatchTST model for time series forecasting.
"""

from pytorch_forecasting.models.patchtst._patchtst_pkg_v2 import PatchTST_pkg_v2
from pytorch_forecasting.models.patchtst._patchtst_v2 import PatchTST

__all__ = ["PatchTST", "PatchTST_pkg_v2"]
