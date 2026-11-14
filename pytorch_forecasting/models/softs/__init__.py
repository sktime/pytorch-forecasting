"""
SOFTS Model for Multivariate Time Series Forecasting.
"""

from pytorch_forecasting.models.softs._softs_forecaster_v2 import SOFTSForecaster
from pytorch_forecasting.models.softs._softs_v2 import SOFTS

__all__ = ["SOFTS", "SOFTSForecaster"]
