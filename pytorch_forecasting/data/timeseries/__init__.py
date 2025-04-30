"""Data loaders for time series data."""

from pytorch_forecasting.data.timeseries._timeseries_v2 import TimeSeries
from pytorch_forecasting.data.timeseries._timeseries import TimeSeriesDataSet

__all__ = [
    "TimeSeriesDataSet",
    "TimeSeries",
]
