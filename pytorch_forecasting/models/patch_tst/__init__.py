"""
PatchTST model for forecasting time series.
"""

from pytorch_forecasting.models.patch_tst._patch_tst_pkg import PatchTST_pkg
from pytorch_forecasting.models.patch_tst._patch_tst_pkg_v2 import PatchTST_pkg_v2
from pytorch_forecasting.models.patch_tst._patch_tst_v2 import PatchTST_v2
from pytorch_forecasting.models.patch_tst.patch_tst import PatchTST

__all__ = [
    "PatchTST",
    "PatchTST_pkg",
    "PatchTST_v2",
    "PatchTST_pkg_v2",
]
