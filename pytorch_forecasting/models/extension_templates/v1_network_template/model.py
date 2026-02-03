"""
Minimal model template for extending PyTorch Forecasting (v1).

This file shows the required structure, not a real working model.
Replace the dummy logic with your own implementation.
"""

from typing import Any, Dict

import torch
from pytorch_forecasting.models import BaseModel


class ExampleNetwork(BaseModel):
    """
    Minimal template model.
    Replace this with your actual neural network.
    """

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        REQUIRED: implement the forward pass.

        Args:
            x: input dictionary from TimeSeriesDataSet

        Returns:
            dictionary with at least a `prediction` key
        """
        # Very simple baseline-style logic (placeholder)
        last_values = x["encoder_target"][:, -1]
        prediction = last_values[:, None]
        return self.to_network_output(prediction=prediction)

    def to_prediction(self, out: Dict[str, Any], use_metric: bool = True, **kwargs):
        """REQUIRED: convert network output to predictions."""
        return out.prediction

    def to_quantiles(self, out: Dict[str, Any], use_metric: bool = True, **kwargs):
        """OPTIONAL: implement if your model supports probabilistic outputs."""
        return out.prediction[..., None]


