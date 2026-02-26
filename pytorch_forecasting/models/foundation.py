from typing import Dict

import torch
from torch import nn

from pytorch_forecasting.models.base._base_model import BaseModel
from pytorch_forecasting.metrics import SMAPE


class FoundationModelWrapper(BaseModel):
    def __init__(self, pretrained_model: nn.Module, loss=SMAPE(), **kwargs):
        self.save_hyperparameters(ignore=["pretrained_model"])
        super().__init__(loss=loss, **kwargs)
        self.pretrained_model = pretrained_model

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoder_input = x["encoder_cont"]
        raw_output = self.pretrained_model(encoder_input)
        return self.to_network_output(prediction=raw_output)
