"""
N-Beats model for timeseries forecasting without covariates.

# TODO v2: remove compatibility imports, kept to avoid breaking existing code.
"""

# Import blocks from new location for backward compatibility
from pytorch_forecasting.layers._nbeats._blocks import (
    NBEATSGenericBlock,
    NBEATSSeasonalBlock,
    NBEATSTrendBlock,
)
from pytorch_forecasting.models.nbeats._grid_callback import GridUpdateCallback
from pytorch_forecasting.models.nbeats._nbeats import NBeats
from pytorch_forecasting.models.nbeats._nbeats_adapter import NBeatsAdapter
from pytorch_forecasting.models.nbeats._nbeats_adapter_v2 import NBeatsAdapterV2
from pytorch_forecasting.models.nbeats._nbeats_pkg import NBeats_pkg
from pytorch_forecasting.models.nbeats._nbeats_pkg_v2 import NBeats_pkg_v2
from pytorch_forecasting.models.nbeats._nbeats_v2 import NBeats_v2
from pytorch_forecasting.models.nbeats._nbeatskan import NBeatsKAN
from pytorch_forecasting.models.nbeats._nbeatskan_pkg import NBeatsKAN_pkg
from pytorch_forecasting.models.nbeats._nbeatskan_pkg_v2 import NBeatsKAN_pkg_v2
from pytorch_forecasting.models.nbeats._nbeatskan_v2 import NBeatsKAN_v2

__all__ = [
    "NBeats",
    "NBeats_v2",
    "NBeats_pkg_v2",
    "NBeatsKAN",
    "NBeatsKAN_v2",
    "NBeatsKAN_pkg_v2",
    "NBeats_pkg",
    "NBeatsKAN_pkg",
    "NBEATSGenericBlock",
    "NBEATSSeasonalBlock",
    "NBEATSTrendBlock",
    "NBeatsAdapter",
    "NBeatsAdapterV2",
    "GridUpdateCallback",
]
