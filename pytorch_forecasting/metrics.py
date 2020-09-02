"""
Implementation of metrics for (mulit-horizon) timeseries forecasting.
"""
from typing import Dict, List, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn
import abc
from pytorch_lightning.metrics import TensorMetric

import scipy.stats


class Metric(TensorMetric):
    """
    Base metric class that has basic functions that can handle predicting quantiles and operate in log space

    Other metrics should inherit from this base class
    """

    def __init__(self, name: str = None, quantiles: List[float] = [0.5], reduction="mean"):
        """
        Initialize metric

        Args:
            name (str): metric name. Defaults to class name.
            quantiles (List[float], optional): quantiles for probability range. Defaults to [0.5].
            reduction (str, optional): Reduction, "none", "mean" or "sqrt-mean". Defaults to "mean".
        """
        self.quantiles = quantiles
        self.reduction = reduction
        if name is None:
            name = self.__class__.__name__
        super().__init__(name)

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
        raise NotImplementedError()

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        if y_pred.ndim == 3:
            idx = self.quantiles.index(0.5)
            y_pred = y_pred[..., idx]
        return y_pred

    def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: prediction quantiles
        """
        if y_pred.ndim == 2:
            y_pred = y_pred.unsqueeze(-1)
        return y_pred


class MultiHorizonMetric(Metric):
    """
    Abstract class for defining metric for a multihorizon forecast
    """

    @abc.abstractmethod
    def loss(self, y_pred: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss without reduction. Override in derived classes

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: loss/metric as a single number for backpropagation
        """
        raise NotImplementedError()

    def forward(self, y_pred: Dict[str, torch.Tensor], target: Union[torch.Tensor, rnn.PackedSequence]) -> torch.Tensor:
        """
        Forward method of metric that handles masking of values.

        Do not override this method but :py:meth:`~loss` instead

        Args:
            y_pred (Dict[str, torch.Tensor]): network output
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
            lengths = torch.ones(target.size(0), device=target.device, dtype=torch.long) * target.size(1)
        assert not target.requires_grad

        # calculate loss with "none" reduction
        if target.ndim == 3:
            weight = target[..., 1]
            target = target[..., 0]
        else:
            weight = None

        losses = self.loss(y_pred, target)
        # weight samples
        if weight is not None:
            losses = losses * weight.unsqueeze(-1)

        # mask loss
        mask = torch.arange(target.size(1), device=target.device).unsqueeze(0) >= lengths.unsqueeze(-1)
        if losses.ndim > 2:
            mask = mask.unsqueeze(-1)
            dim_normalizer = losses.size(-1)
        else:
            dim_normalizer = 1.0
        # reduce to one number
        if self.reduction == "none":
            loss = losses.masked_fill(mask, float("nan"))
        else:
            if self.reduction == "mean":
                losses = losses.masked_fill(mask, 0.0)
                loss = losses.sum() / lengths.sum() / dim_normalizer
            elif self.reduction == "sqrt-mean":
                losses = losses.masked_fill(mask, 0.0)
                loss = losses.sum() / lengths.sum() / dim_normalizer
                loss = loss.sqrt()
            assert not torch.isnan(loss), (
                "Loss should not be nan - i.e. something went wrong "
                "in calculating the loss (e.g. log of a negative number)"
            )
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

    def loss(self, y_pred: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        return F.poisson_nll_loss(
            super().to_prediction(y_pred), target, log_input=True, full=False, eps=1e-6, reduction="none"
        )

    def to_prediction(self, out: Dict[str, torch.Tensor]):
        rate = torch.exp(super().to_prediction(out))
        return rate

    def to_quantiles(self, out: Dict[str, torch.Tensor], quantiles=None):
        if quantiles is None:
            quantiles = self.quantiles
        return scipy.stats.poisson(super().to_prediction(out)).ppf(quantiles)


class QuantileLoss(MultiHorizonMetric):
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calcualted as

    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """

    def __init__(
        self,
        name: str = "quantile_loss",
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        *args,
        **kwargs
    ):
        """
        Quantile loss

        Args:
            name: name of metric
        """
        super().__init__(name, quantiles=quantiles, *args, **kwargs)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate quantile loss
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = torch.cat(losses, dim=2)

        return losses


class SMAPE(MultiHorizonMetric):
    """
    Symmetric mean average percentage. Assumes ``y >= 0``.

    Defined as ``2*(y - y_pred).abs() / (y.abs() + y_pred.abs())``
    """

    def __init__(self, name: str = "sMAPE", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def loss(self, y_pred, target):
        y_pred = self.to_prediction(y_pred)
        loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
        return loss


class MAPE(MultiHorizonMetric):
    """
    Mean average percentage. Assumes ``y >= 0``.

    Defined as ``(y - y_pred).abs() / y.abs()``
    """

    def __init__(self, name: str = "MAPE", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def loss(self, y_pred, target):
        loss = (self.to_prediction(y_pred) - target).abs() / (target.abs() + 1e-8)
        return loss


class MAE(MultiHorizonMetric):
    """
    Mean average absolute error.

    Defined as ``(y_pred - target).abs()``
    """

    def __init__(self, name: str = "MAE", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def loss(self, y_pred, target):
        loss = (self.to_prediction(y_pred) - target).abs()
        return loss


class RMSE(MultiHorizonMetric):
    """
    Root mean square error

    Defined as ``(y_pred - target)**2``
    """

    def __init__(self, name: str = "RMSE", reduction="sqrt-mean", *args, **kwargs):
        super().__init__(name, *args, reduction=reduction, **kwargs)

    def loss(self, y_pred: Dict[str, torch.Tensor], target):
        loss = torch.pow(self.to_prediction(y_pred) - target, 2)
        return loss
