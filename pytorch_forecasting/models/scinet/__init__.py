"""SCINet v2 model for time series forecasting."""

from pytorch_forecasting.models.scinet._scinet_pkg_v2 import SCINet_v2_pkg_v2
from pytorch_forecasting.models.scinet._scinet_v2 import SCINet_v2

__all__ = ["SCINet_v2", "SCINet_v2_pkg_v2"]
