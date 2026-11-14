"""
N-BEATS models, including NBEATSx for forecasting with covariates.

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
from pytorch_forecasting.models.nbeats._nbeats_pkg import NBeats_pkg
from pytorch_forecasting.models.nbeats._nbeatskan import NBeatsKAN
from pytorch_forecasting.models.nbeats._nbeatskan_pkg import NBeatsKAN_pkg
from pytorch_forecasting.models.nbeats._nbeatsx import NBeatsX
from pytorch_forecasting.models.nbeats._nbeatsx_pkg import NBeatsX_pkg

__all__ = [
    "NBeats",
    "NBeatsKAN",
    "NBeatsX",
    "NBeatsX_pkg",
    "NBeats_pkg",
    "NBeatsKAN_pkg",
    "NBEATSGenericBlock",
    "NBEATSSeasonalBlock",
    "NBEATSTrendBlock",
    "NBeatsAdapter",
    "GridUpdateCallback",
]
