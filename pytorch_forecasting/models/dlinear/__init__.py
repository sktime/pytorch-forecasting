"""
Decomposition-Linear model for time series forecasting.
"""

from pytorch_forecasting.models.dlinear._dlinear_forecaster_v2 import DLinearForecaster
from pytorch_forecasting.models.dlinear._dlinear_v2 import DLinear

__all__ = ["DLinear", "DLinearForecaster"]
