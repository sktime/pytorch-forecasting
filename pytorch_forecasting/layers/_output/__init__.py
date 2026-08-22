"""
Implementation of output layers for PyTorch Forecasting.
"""

from pytorch_forecasting.layers._output._flatten_head import (
    FlattenHead,
)
from pytorch_forecasting.layers._output._patch_tst_flatten_head import (
    PatchTSTFlattenHead,
)

__all__ = ["FlattenHead", "PatchTSTFlattenHead"]
