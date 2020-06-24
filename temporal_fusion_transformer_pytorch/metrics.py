from typing import List, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn
import abc
from pytorch_lightning.metrics import TensorMetric

import scipy.stats


class MultiHorizonMetric(TensorMetric, metaclass=abc.ABCMeta):
    """
    Abstract class for defining metric
    """

    def __init__(self, name: str, quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98], *args, **kwargs):
        self.quantiles = quantiles
        super().__init__(name, *args, **kwargs)

    @abc.abstractmethod
    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        calculate loss without reduction
        """
        pass

    def forward(self, y_pred: torch.Tensor, target: Union[torch.Tensor, rnn.PackedSequence]) -> torch.Tensor:
        # unpack
        if isinstance(target, rnn.PackedSequence):
            target, lengths = rnn.pad_packed_sequence(target, batch_first=True)
        else:
            lengths = torch.LongTensor([target.size(1)]).expand(target.size(0))
        assert not target.requires_grad
        assert y_pred.size(0) == target.size(0)

        # calculate loss with "none" reduction
        losses = self.loss(y_pred.squeeze(), target)

        # mask loss
        mask = torch.arange(target.size(1)).unsqueeze(0) >= lengths.unsqueeze(-1)
        if self.input_size > 1:
            mask = mask.unsqueeze(-1)
        losses = losses.masked_fill(mask, 0.0)

        # reduce to one number
        loss = losses.sum() / lengths.sum()
        return loss

    @property
    @abc.abstractmethod
    def input_size(self) -> int:
        """
        number of dimensions of prediction (e.g. 5 for 5 different quantiles)
        """
        return 1

    @abc.abstractmethod
    def to_prediction(self, out: torch.Tensor) -> torch.Tensor:
        return out

    @abc.abstractmethod
    def to_quantiles(self, out: torch.Tensor):
        return out


class PoissonLoss(MultiHorizonMetric):
    def __init__(self, *args, **kwargs):
        return super().__init__(name="poisson_loss", *args, **kwargs)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.poisson_nll_loss(y_pred, target, log_input=True, full=False, eps=1e-6, reduction="none")

    @property
    def input_size(self) -> int:
        return 1

    def to_prediction(self, out):
        rate = torch.exp(out[..., 0])
        return rate

    def to_quantiles(self, out, quantiles=None):
        if quantiles is None:
            quantiles = self.quantiles
        return scipy.stats.poisson(self.to_prediction(out).unsqueeze(-1)).ppf(quantiles)


class QuantileLoss(MultiHorizonMetric):
    def __init__(self, log_space: bool = None, *args, **kwargs):
        """
        Quantile loss

        Args:
            log_space: if model should be estimated in log
            cummulative (bool): if loss should be calculated cummulatively, i.e.
                if false, the quantiles hold true for individual predictions but
                if true, the quantiles hold true if the predictions are cummulatively
                summed. This is useful if total quantities over the prediction horizon
                are supposed to be predicted.
        """
        super().__init__(name="quantile_loss", *args, **kwargs)
        self.log_space = log_space

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # if self.cummulative:
        #     if self.log_target:
        #         preds = preds.exp().cumsum(dim=-2).log()
        #         target = target.exp().cumsum(dim=-1).log()
        #     else:
        #         preds = preds.cumsum(dim=-2)
        #         target = target.cumsum(dim=-1)
        if self.log_space:
            target = torch.log(target + 1e-6)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = torch.cat(losses, dim=2)
        return losses

    @property
    def input_size(self) -> int:
        return len(self.quantiles)

    def to_quantiles(self, out):
        if self.log_space:
            out = out.exp()
        return out

    def to_prediction(self, out):
        pred = out[..., self.quantiles.index(0.5)]
        if self.log_space:
            pred = pred.exp()
        return pred
