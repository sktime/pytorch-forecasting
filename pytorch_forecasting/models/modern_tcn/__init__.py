"""Modern Temporal Convolutional Network model for time series forecasting."""

from pytorch_forecasting.models.modern_tcn._modern_tcn_forecaster_v2 import (
    ModernTCNForecaster,
)
from pytorch_forecasting.models.modern_tcn._modern_tcn_v2 import ModernTCN

__all__ = ["ModernTCN", "ModernTCNForecaster"]
