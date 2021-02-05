"""
Baseline model.
"""
from typing import Dict

import torch
from torch.nn.utils import rnn

from pytorch_forecasting.metrics import MultiHorizonMetric, QuantileLoss
from pytorch_forecasting.models import BaseModel


class Baseline(BaseModel):
    """
    Baseline model that uses last known target value to make prediction.
    """

    def __init__(self, output_size: int = 7, loss: MultiHorizonMetric = QuantileLoss()):
        self.save_hyperparameters()
        super().__init__()
        self.loss = loss

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Network forward pass.

        Args:
            x (Dict[str, torch.Tensor]): network input

        Returns:
            Dict[str, torch.Tensor]: netowrk outputs
        """
        max_prediction_length = x["decoder_lengths"].max()
        assert x["encoder_lengths"].min() > 0, "Encoder lengths of at least 1 required to obtain last value"
        last_values = x["encoder_target"][torch.arange(x["encoder_target"].size(0)), x["encoder_lengths"] - 1]
        prediction = last_values[:, None, None].expand(-1, max_prediction_length, self.hparams.output_size)
        return dict(prediction=prediction)

    def _step(self, batch, batch_idx):
        """
        run at each step for training or validation
        """
        # extract data and run model
        x, y = batch
        y = rnn.pack_padded_sequence(y, lengths=x["decoder_lengths"], batch_first=True, enforce_sorted=False)
        log, _ = super()._step(x, y, batch_idx)
        return log
