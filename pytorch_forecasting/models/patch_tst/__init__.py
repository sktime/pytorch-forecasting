"""
PatchTST model for forecasting time series.
"""

from pytorch_forecasting.models.patch_tst.patch_tst import PatchTST
from pytorch_forecasting.models.patch_tst._patch_tst_v2 import PatchTST as PatchTSTV2

__all__ = [
    "PatchTST",
    "PatchTSTV2",
]
