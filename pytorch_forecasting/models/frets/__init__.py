"""FreTS v2 model for time series forecasting."""

from pytorch_forecasting.models.frets._frets_forecaster_v2 import FreTSForecaster
from pytorch_forecasting.models.frets._frets_v2 import FreTS

__all__ = ["FreTS", "FreTSForecaster"]
