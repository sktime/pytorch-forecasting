"""
Backward-compatibility shim for N-BEATS blocks.
Real implementations live in `ptf.layers._nbeats._blocks`.

# TODO v2: remove this file.
"""

from ptf.layers._nbeats._blocks import (
    NBEATSGenericBlock,
    NBEATSSeasonalBlock,
    NBEATSTrendBlock,
)
