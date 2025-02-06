"""Tide model."""

from pytorch_forecasting.models.tide._tide import TiDEModel
from pytorch_forecasting.models.tide.sub_modules import _TideModule

__all__ = [
    "_TideModule",
    "TiDEModel",
]
