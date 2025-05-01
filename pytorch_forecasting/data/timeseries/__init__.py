"""Data loaders for time series data."""

from pytorch_forecasting.data.timeseries._timeseries import TimeSeriesDataSet
from pytorch_forecasting.data.timeseries._timeseries_v2 import TimeSeries

__all__ = [
    "TimeSeriesDataSet",
    "TimeSeries",
]
