"""
Backward-compatibility shim for N-BEATS blocks.
Real implementations live in `pytorch_forecasting.layers._nbeats._blocks`.

# TODO v2: remove this file.
"""

from pytorch_forecasting.layers._nbeats._blocks import (
    NBEATSGenericBlock,
    NBEATSSeasonalBlock,
    NBEATSTrendBlock,
)
