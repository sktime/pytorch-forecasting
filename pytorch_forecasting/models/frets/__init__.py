"""FreTS v2 model for time series forecasting."""

from pytorch_forecasting.models.frets._frets_pkg_v2 import FreTS_v2_pkg_v2
from pytorch_forecasting.models.frets._frets_v2 import FreTS_v2

__all__ = ["FreTS_v2", "FreTS_v2_pkg_v2"]
