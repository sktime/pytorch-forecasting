"""SCINet v2 model for time series forecasting."""

from pytorch_forecasting.models.scinet._scinet_pkg_v2 import SCINet_pkg_v2
from pytorch_forecasting.models.scinet._scinet_v2 import SCINet

__all__ = ["SCINet", "SCINet_pkg_v2"]
