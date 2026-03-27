"""
V2 Model Template - Consolidated.
RATIONALE: Provides a canonical starting point with minimal boilerplate.
"""

import torch
import torch.nn as nn

from pytorch_forecasting.base._base_pkg import Base_pkg
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class MyNewModel(BaseModel):
    """
    CONTRIBUTOR TODO: Implement __init__ and forward.
    """

    @classmethod
    def _pkg(cls):
        return MyNewModel_pkg

    def __init__(self, metadata, loss, hidden_size=64, **kwargs):
        super().__init__(loss=loss, **kwargs)
        # RATIONALE: Initialize your network here.
        # self.network = nn.Linear(metadata["encoder_cont"], hidden_size)

    def forward(self, x):
        """
        RATIONALE: Primary forward pass logic.
        """
        # prediction = self.network(x["encoder_cont"])
        # return {"prediction": prediction}
        raise NotImplementedError("Implement forward logic.")

    def training_step(self, batch, batch_idx):
        """
        Optional: Override if you need custom logic (e.g., teacher forcing).
        """
        return super().training_step(batch, batch_idx)


class MyNewModel_pkg(Base_pkg):
    """
    Metadata and testing configuration.
    """

    _tags = {
        "info:name": "MyNewModel",
        "object_type": "model",
    }

    @classmethod
    def get_cls(cls):
        return MyNewModel

    @classmethod
    def get_datamodule_cls(cls):
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    @classmethod
    def get_test_train_params(cls):
        """
        RATIONALE: Provides parameter sets for automated testing.
        """
        return [dict(hidden_size=16, loss=torch.nn.MSELoss())]
