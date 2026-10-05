"""
Prototype of the reworked ``pytorch-forecasting`` v2 API.
"""

from pytorch_forecasting._proto._base_forecaster import BaseForecaster
from pytorch_forecasting._proto._tft_forecaster import TFTForecaster
from pytorch_forecasting._proto._timeseries_datatype import (
    TimeSeries_datatype,
    TimeSeriesMetadata,
)

__all__ = [
    "BaseForecaster",
    "TFTForecaster",
    "TimeSeriesMetadata",
    "TimeSeries_datatype",
]
