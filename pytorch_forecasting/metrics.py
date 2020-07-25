from abc import abstractmethod
from typing import List, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn
import abc
from pytorch_lightning.metrics import TensorMetric

import scipy.stats


class Metric(TensorMetric, metaclass=abc.ABCMeta):
    """
    Base metric class that has basic functions that can handle predicting quantiles and operate in log space

    Other metrics should inherit from this base class
    """

    def __init__(
        self, name: str, log_space: bool = False, quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    ):
        self.log_space = log_space
        self.quantiles = quantiles
        super().__init__(name)

    @abstractmethod
    def forward(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        Abstract method that calcualtes metric

        Should be overriden in derived classes

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        pass

    def to_prediction(self, out: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            out (torch.Tensor): output of network

        Returns:
            torch.Tensor: point prediction
        """
        if self.log_space:
            out = out.exp()
        return out

    def to_quantiles(self, out: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            out (torch.Tensor): output of network

        Returns:
            torch.Tensor: prediction quantiles
        """
        if self.log_space:
            out = out.exp()
        return out.unsqueeze(1)


class MultiHorizonMetric(Metric):
    """
    Abstract class for defining metric for a multihorizon forecast
    """

    def __init__(self, name="loss", cummulative=False, *args, **kwargs):
        """
        Initialize multi-horizon loss

        Args:
            cummulative: if loss should be calculated cummulatively, i.e.
                if false, the quantiles hold true for individual predictions but
                if true, the quantiles hold true if the predictions are cummulatively
                summed. This is useful if total quantities over the prediction horizon
                are supposed to be predicted.
        """
        super().__init__(name, *args, **kwargs)
        self.cummulative = cummulative

    @abc.abstractmethod
    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss without reduction. Override in derived classes

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: loss/metric as a single number for backpropagation
        """
        pass

    def forward(self, y_pred: torch.Tensor, target: Union[torch.Tensor, rnn.PackedSequence]) -> torch.Tensor:
        """
        Forward method of metric that handles masking of values.

        Do not override this method but :py:ref:`~loss` instead

        Args:
            y_pred (torch.Tensor): network output
            target (Union[torch.Tensor, rnn.PackedSequence]): actual values

        Returns:
            torch.Tensor: loss as a single number for backpropagation
        """
        # unpack
        if isinstance(target, rnn.PackedSequence):
            target, lengths = rnn.pad_packed_sequence(target, batch_first=True)
            # batch sizes reside on the CPU by default -> we need to bring them to GPU
            lengths = lengths.to(target.device)
        else:
            lengths = torch.LongTensor([target.size(1)], device=target.device).expand(target.size(0))
        assert not target.requires_grad
        assert y_pred.size(0) == target.size(0)

        # calculate loss with "none" reduction
        if target.ndim == 3:
            weight = target[..., 1]
            target = target[..., 0]
        else:
            weight = None

        # prepare for cummulative
        if self.cummulative:
            if self.log_space:
                y_pred = y_pred.exp().cumsum(dim=-2).log()
            else:
                y_pred = y_pred.cumsum(dim=-2)
            target = (target.cumsum(dim=-1) + 1e-8).log()
        else:
            # transform prediction into normal space
            if self.log_space:
                target = (target + 1e-8).log()

        losses = self.loss(y_pred, target)
        # weight samples
        if weight is not None:
            losses = losses * weight.unsqueeze(-1)

        # mask loss
        mask = torch.arange(target.size(1), device=target.device).unsqueeze(0) >= lengths.unsqueeze(-1)
        if y_pred.ndim > 2:
            mask = mask.unsqueeze(-1)
        losses = losses.masked_fill(mask, 0.0)

        # reduce to one number
        loss = losses.sum() / lengths.sum()
        assert not torch.isnan(
            loss
        ), "Loss should not be nan - i.e. something went wrong in calculating the loss (e.g. log of a negative number)"
        assert torch.isfinite(
            loss
        ), "Loss should not be infinite - i.e. something went wrong (e.g. input is not in log space)"
        return loss


class PoissonLoss(MultiHorizonMetric):
    """
    Poisson loss for count data
    """

    def __init__(self, name: str = "poisson_loss", *args, **kwargs):
        return super().__init__(name, *args, **kwargs)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 3:
            raise NotImplementedError("Weights are not supported for Poisson loss")
        return F.poisson_nll_loss(y_pred.squeeze(2), target, log_input=True, full=False, eps=1e-6, reduction="none")

    def to_prediction(self, out):
        rate = torch.exp(out[..., 0])
        return rate

    def to_quantiles(self, out, quantiles=None):
        if quantiles is None:
            quantiles = self.quantiles
        return scipy.stats.poisson(self.to_prediction(out).unsqueeze(-1)).ppf(quantiles)


class QuantileLoss(MultiHorizonMetric):
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calcualted as

    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """

    def __init__(
        self,
        name: str = "quantile_loss",
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        log_space: bool = None,
        cummulative=False,
        *args,
        **kwargs
    ):
        """
        Quantile loss

        Args:
            name: name of metric
            log_space: if model should be estimated in log
            cummulative: if loss should be calculated cummulatively, i.e.
                if false, the quantiles hold true for individual predictions but
                if true, the quantiles hold true if the predictions are cummulatively
                summed. This is useful if total quantities over the prediction horizon
                are supposed to be predicted.
        """
        super().__init__(name, log_space=log_space, quantiles=quantiles, cummulative=cummulative, *args, **kwargs)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate quantile loss
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = torch.cat(losses, dim=2)

        return losses

    def to_quantiles(self, out):
        if self.log_space:
            out = out.exp()
        return out

    def to_prediction(self, out):
        pred = out[..., self.quantiles.index(0.5)]
        if self.log_space:
            pred = pred.exp()
        return pred


class SMAPE(MultiHorizonMetric):
    """
    Symmetric mean average percentage. Assumes ``y >= 0``.

    Defined as ``(y - y_pred).abs() / (y.abs() + y_pred.abs())``
    """

    def __init__(self, name: str = "sMAPE", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def loss(self, y_pred, target):
        loss = (y_pred - target).abs() / (y_pred.abs() + target.abs())
        return loss


class MAPE(MultiHorizonMetric):
    """
    Mean average percentage. Assumes ``y >= 0``.

    Defined as ``(y - y_pred).abs() / y.abs()``
    """

    def __init__(self, name: str = "MAPE", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def loss(self, y_pred, target):
        loss = (y_pred - target).abs() / (target.abs() + 1e-8)
        return loss


class MAE(MultiHorizonMetric):
    """
    Mean average absolute error.

    Defined as ``(y_pred - target).abs()``
    """

    def __init__(self, name: str = "MAE", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def loss(self, y_pred, target):
        loss = (y_pred - target).abs()
        return loss


class RMSE(MultiHorizonMetric):
    """
    Root mean square error

    Defined as ``(y_pred - target)**2``
    """

    def __init__(self, name: str = "RMSE", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def loss(self, y_pred, target):
        loss = torch.pow(y_pred - target, 2)
        return loss
