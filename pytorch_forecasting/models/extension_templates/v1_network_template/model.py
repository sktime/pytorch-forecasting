"""
Minimal Lightning model template for extending PyTorch Forecasting (v1).
"""

from typing import Any

import torch

from pytorch_forecasting.models import BaseModel


class ExampleNetwork(BaseModel):
    """
    Minimal template model for contributors.
    """

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Minimal placeholder forward pass.
        Users should replace this with their own implementation.
        """
        last_values = x["encoder_target"][:, -1]
        prediction = last_values[:, None]
        return self.to_network_output(prediction=prediction)

    def to_prediction(self, out: dict[str, Any], use_metric: bool = True, **kwargs):
        return out.prediction

    def to_quantiles(self, out: dict[str, Any], use_metric: bool = True, **kwargs):
        return out.prediction[..., None]
