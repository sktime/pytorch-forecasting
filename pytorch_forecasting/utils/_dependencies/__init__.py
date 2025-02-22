"""Utilities for managing dependencies."""

from pytorch_forecasting.utils._dependencies._dependencies import (
    _check_matplotlib,
    _get_installed_packages,
)
from pytorch_forecasting.utils._dependencies._safe_import import _safe_import

__all__ = [
    "_get_installed_packages",
    "_check_matplotlib",
    "_safe_import",
]
