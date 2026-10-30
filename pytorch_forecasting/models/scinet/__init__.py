"""SCINet v2 model for time series forecasting."""

from pytorch_forecasting.models.scinet._scinet_forecaster_v2 import SCINetForecaster
from pytorch_forecasting.models.scinet._scinet_v2 import SCINet_v2

__all__ = ["SCINet_v2", "SCINetForecaster"]
