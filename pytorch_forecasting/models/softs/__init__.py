"""
SOFTS Model for Multivariate Time Series Forecasting.
"""

from pytorch_forecasting.models.softs._softs_pkg_v2 import Softs_pkg_v2
from pytorch_forecasting.models.softs._softs_v2 import Softs

__all__ = ["Softs", "Softs_pkg_v2"]
