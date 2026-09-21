"""FreTS v2 model for time series forecasting."""

from pytorch_forecasting.models.frets._frets_pkg_v2 import FreTS_pkg_v2
from pytorch_forecasting.models.frets._frets_v2 import FreTS

__all__ = ["FreTS", "FreTS_pkg_v2"]
