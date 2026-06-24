"""
SOFTS Model for Multivariate Time Series Forecasting.
"""

from pytorch_forecasting.models.softs._softs_pkg_v2 import SOFTS_pkg_v2
from pytorch_forecasting.models.softs._softs_v2 import SOFTS

__all__ = ["SOFTS", "SOFTS_pkg_v2"]
