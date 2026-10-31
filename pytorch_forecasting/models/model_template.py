"""
Template for implementing a model under v2 architecture.
"""

from typing import Any

import torch
import torch.nn as nn

from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class MyNewModel(BaseModel):
    """
    CONTRIBUTOR TODO: Implement your model logic here.

    Defines architecture and forward pass. Inherits standard training logic.
    """

    @classmethod
    def _pkg(cls):
        """
        Link to the model's package container.
        """
        from pytorch_forecasting.models.pkg_template import MyNewModel_pkg

        return MyNewModel_pkg

    def __init__(
        self, metadata: dict[str, Any], loss: nn.Module, hidden_size: int = 64, **kwargs
    ):
        """
        Initialize the model.

        Parameters:
        -----------
        metadata : dict[str, Any]
            Input data metadata. Expected keys: "encoder_cont",
            "encoder_cat", "max_prediction_length", etc.
        loss : nn.Module
            Loss function for training.
        hidden_size : int, default=64
            Example hyperparameter.
        **kwargs : Any
            Additional arguments for BaseModel.
        """
        super().__init__(loss=loss, **kwargs)

        # RATIONALE: Initialize layers.
        # self.network = nn.Linear(metadata["encoder_cont"], hidden_size)

        self.save_hyperparameters()

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Primary forward pass.

        Parameters:
        -----------
        x : dict[str, torch.Tensor]
            Input tensors (e.g., "encoder_cont", "y", "group").

        Returns:
        --------
        dict[str, torch.Tensor]
            Output containing "prediction".
        """
        # CONTRIBUTOR TODO: Implement logic.
        # prediction = self.network(x["encoder_cont"])
        # return {"prediction": prediction}
        raise NotImplementedError("Implement forward logic.")

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Individual training step.

        Optional: Override for custom logic (e.g., teacher forcing).
        """
        return super().training_step(batch, batch_idx)
