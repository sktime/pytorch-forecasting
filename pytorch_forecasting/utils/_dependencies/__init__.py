"""Utilities for managing dependencies."""

from pytorch_forecasting.utils._dependencies._dependencies import _check_matplotlib
from pytorch_forecasting.utils._dependencies._safe_import import _safe_import

__all__ = [
    "_check_matplotlib",
    "_safe_import",
]
