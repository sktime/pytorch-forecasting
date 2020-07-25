from typing import Dict
import torch
from torch.nn.utils import rnn

from pytorch_forecasting.models import BaseModel
from pytorch_forecasting.metrics import MultiHorizonMetric, QuantileLoss


class Baseline(BaseModel):
    """
    Baseline model that uses last value to make prediction.
    """

    def __init__(self, output_size: int = 7, loss: MultiHorizonMetric = QuantileLoss()):
        self.save_hyperparameters()
        super().__init__()
        self.loss = loss

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        max_prediction_length = x["decoder_lengths"].max()
        assert x["encoder_lengths"].min() > 0, "Encoder lengths of at least 1 required to obtain last value"
        last_values = x["encoder_target"][torch.arange(x["encoder_target"].size(0)), x["encoder_lengths"] - 1]
        return last_values[:, None, None].expand(-1, max_prediction_length, self.hparam.output_size)

    def _step(self, batch, batch_idx, label="train"):
        """
        run at each step for training or validation
        """
        # extract data and run model
        x, y = batch
        y = rnn.pack_padded_sequence(y, lengths=x["decoder_lengths"], batch_first=True, enforce_sorted=False)
        log, _ = super()._step(x, y, batch_idx, label=label)
        return log
