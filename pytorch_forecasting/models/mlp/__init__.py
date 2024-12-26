"""Simple models based on fully connected networks."""

from pytorch_forecasting.models.mlp._decodermlp import DecoderMLP
from pytorch_forecasting.models.mlp.submodules import FullyConnectedModule

__all__ = ["DecoderMLP", "FullyConnectedModule"]
