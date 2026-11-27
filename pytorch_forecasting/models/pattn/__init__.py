"""
PAttn model for time series forecasting.
"""

from pytorch_forecasting.models.pattn._pattn_forecaster_v2 import PAttnForecaster
from pytorch_forecasting.models.pattn._pattn_v2 import PAttn

__all__ = ["PAttn", "PAttnForecaster"]
