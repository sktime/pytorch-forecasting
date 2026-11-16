"""Baseline forecaster model."""

from pytorch_forecasting.models.baseline._baseline import Baseline
from pytorch_forecasting.models.baseline._baseline_pkg_v2 import Baseline_pkg_v2
from pytorch_forecasting.models.baseline._baseline_v2 import Baseline_v2

__all__ = ["Baseline", "Baseline_v2", "Baseline_pkg_v2"]
