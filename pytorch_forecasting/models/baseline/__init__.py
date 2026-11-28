"""baseline model"""

from pytorch_forecasting.models.baseline._baseline import Baseline
from pytorch_forecasting.models.baseline._naive_forecaster_v2 import NaiveForecaster
from pytorch_forecasting.models.baseline._naive_v2 import Naive

__all__ = ["Baseline", "Naive", "NaiveForecaster"]
