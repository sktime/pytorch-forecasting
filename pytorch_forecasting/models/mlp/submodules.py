"""
Backward-compatibility shim for the fully connected MLP module.
Real implementation lives in `pytorch_forecasting.layers._mlp`.

# TODO v2: remove this file.
"""

from pytorch_forecasting.layers._mlp import FullyConnectedModule  # noqa: F401
