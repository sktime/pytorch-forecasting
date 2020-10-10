"""
Implementation of metrics for (mulit-horizon) timeseries forecasting.
"""
from typing import Dict, List, Union
import warnings

from pytorch_lightning.metrics import Metric as LightningMetric
import scipy.stats
import torch
import torch.nn.functional as F
from torch.nn.utils import rnn

from pytorch_forecasting.utils import integer_histogram, unpack_sequence


class Metric(LightningMetric):
    """
    Base metric class that has basic functions that can handle predicting quantiles and operate in log space.
    See the `Lightning documentation <https://pytorch-lightning.readthedocs.io/en/latest/metrics.html>`_
    for details of how to implement a new metric

    Other metrics should inherit from this base class
    """

    def __init__(self, name: str = None, quantiles: List[float] = None, reduction="mean"):
        """
        Initialize metric

        Args:
            name (str): metric name. Defaults to class name.
            quantiles (List[float], optional): quantiles for probability range. Defaults to None.
            reduction (str, optional): Reduction, "none", "mean" or "sqrt-mean". Defaults to "mean".
        """
        self.quantiles = quantiles
        self.reduction = reduction
        if name is None:
            name = self.__class__.__name__
        self.name = name
        super().__init__()

    def update(y_pred: torch.Tensor, y_actual: torch.Tensor):
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
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
            if self.quantiles is None:
                idx = y_pred.size(-1) // 2
            else:
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

    def __add__(self, metric: LightningMetric):
        composite_metric = CompositeMetric(metrics=[self])
        new_metric = composite_metric + metric
        return new_metric

    def __mul__(self, multiplier: float):
        new_metric = CompositeMetric(metrics=[self], weights=[multiplier])
        return new_metric

    __rmul__ = __mul__


class CompositeMetric(LightningMetric):
    """
    Metric that combines multiple metrics.

    Metric does not have to be called explicitly but is automatically created when adding and multiplying metrics
    with each other.

    Example:

        .. code-block:: python

            composite_metric = SMAPE() + 0.4 * MAE()
    """

    def __init__(self, metrics: List[LightningMetric] = [], weights: List[float] = None):
        """
        Args:
            name (str, optional): name of composite metric. Defaults to "composite".
            metrics (List[LightningMetric], optional): list of metrics to combine. Defaults to [].
            weights (List[float], optional): list of weights / multipliers for weights. Defaults to 1.0 for all metrics.
        """
        if weights is None:
            weights = [1.0 for _ in metrics]
        assert len(weights) == len(metrics), "Number of weights has to match number of metrics"

        self.metrics = metrics
        self.weights = weights

        super().__init__()

    def __repr__(self):
        name = " + ".join([f"{w:.3g} * {repr(m)}" if w != 1.0 else repr(m) for w, m in zip(self.weights, self.metrics)])
        return name

    def update(self, y_pred: torch.Tensor, y_actual: torch.Tensor):
        """
        Update composite metric

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        for metric in self.metrics:
            metric.update(y_pred, y_actual)

    def compute(self) -> torch.Tensor:
        """
        Get metric

        Returns:
            torch.Tensor: metric
        """
        results = []
        for weight, metric in zip(self.weights, self.metrics):
            results.append(metric.compute() * weight)

        if len(results) == 1:
            results = results[0]
        else:
            results = torch.stack(results, dim=0).sum(0)
        return results

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Will use first metric in ``metrics`` attribute to calculate result.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        return self.metrics[0].to_prediction(y_pred)

    def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Will use first metric in ``metrics`` attribute to calculate result.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: prediction quantiles
        """
        return self.metrics[0].to_quantiles(y_pred)

    def __add__(self, metric: LightningMetric):
        if isinstance(metric, self.__class__):
            self.metrics.extend(metric.metrics)
            self.weights.extend(metric.weights)
        else:
            self.metrics.append(metric)
            self.weights.append(1.0)

        return self

    def __mul__(self, multiplier: float):
        self.weights = [w * multiplier for w in self.weights]
        return self

    __rmul__ = __mul__


class AggregationMetric(Metric):
    """
    Calculate metric on mean prediction and actuals.
    """

    def __init__(self, metric: Metric, **kwargs):
        """
        Args:
            metric (Metric): metric which to calculate on aggreation.
        """
        super().__init__(**kwargs)
        self.metric = metric

    def update(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        Calculate composite metric

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        y_pred_mean = y_pred.mean(0).unsqueeze(0)
        if isinstance(y_actual, rnn.PackedSequence):
            target, lengths = rnn.pad_packed_sequence(y_actual, batch_first=True)
            # batch sizes reside on the CPU by default -> we need to bring them to GPU
            lengths = lengths.to(target.device)

            # calculate mean for all time steps
            tmask = torch.arange(target.size(1), device=target.device).unsqueeze(0) >= lengths.unsqueeze(-1)
            if target.ndim > 2:
                tmask = tmask.unsqueeze(-1)
                lengths = lengths.unsqueeze(-1)
            target = target.masked_fill(tmask, 0.0)
            y_mean = target.sum(0).unsqueeze(0) / lengths.sum()

            # calculate weight as length
            decoder_length_histogram = integer_histogram(lengths, min=1, max=target.size(1))
            weight = decoder_length_histogram.flip(0).cumsum(0).flip(0).float().unsqueeze(0)

            # modify weight
            if y_mean.ndim == 3:
                y_mean[..., 1] = y_mean[..., 1] * weight
            else:
                y_mean = torch.stack((y_mean, weight), dim=-1)

        else:
            y_mean = y_actual.mean(0).unsqueeze(0)
        self.metric.update(y_pred_mean, y_mean)

    def compute(self):
        return self.metric.compute()


class MultiHorizonMetric(Metric):
    """
    Abstract class for defining metric for a multihorizon forecast
    """

    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        super().__init__(reduction=reduction, **kwargs)
        self.add_state("losses", default=torch.tensor(0.0), dist_reduce_fx="sum" if reduction != "none" else "cat")
        self.add_state("lengths", default=torch.tensor(0), dist_reduce_fx="sum" if reduction != "none" else "mean")

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

    def update(self, y_pred: Dict[str, torch.Tensor], target: Union[torch.Tensor, rnn.PackedSequence]):
        """
        Update method of metric that handles masking of values.

        Do not override this method but :py:meth:`~loss` instead

        Args:
            y_pred (Dict[str, torch.Tensor]): network output
            target (Union[torch.Tensor, rnn.PackedSequence]): actual values

        Returns:
            torch.Tensor: loss as a single number for backpropagation
        """
        target, lengths = unpack_sequence(target)
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
        self._update_losses_and_lengths(losses, lengths)

    def _update_losses_and_lengths(self, losses: torch.Tensor, lengths: torch.Tensor):
        losses = self.mask_losses(losses, lengths)
        if self.reduction == "none":
            if self.losses.ndim == 0:
                self.losses = losses
                self.lengths = self.lengths
            else:
                self.losses = torch.cat([self.losses, losses], dim=0)
                self.lengths = torch.cat([self.lengths, lengths], dim=0)
        else:
            losses = losses.sum()
            if not torch.isfinite(losses):
                losses = torch.tensor(1e9)
                warnings.warn("Loss is not finite. Resetting it to 1e9")
            self.losses = self.losses + losses
            self.lengths = self.lengths + lengths.sum()

    def compute(self):
        loss = self.reduce_loss(self.losses, lengths=self.lengths)
        return loss

    def mask_losses(self, losses: torch.Tensor, lengths: torch.Tensor, reduction: str = None) -> torch.Tensor:
        """
        Mask losses.

        Args:
            losses (torch.Tensor): total loss. first dimenion are samples, second timesteps
            lengths (torch.Tensor): total length
            reduction (str, optional): type of reduction. Defaults to ``self.reduction``.

        Returns:
            torch.Tensor: masked losses
        """
        if reduction is None:
            reduction = self.reduction
        if losses.ndim > 0:
            # mask loss
            mask = torch.arange(losses.size(1), device=losses.device).unsqueeze(0) >= lengths.unsqueeze(-1)
            if losses.ndim > 2:
                mask = mask.unsqueeze(-1)
                dim_normalizer = losses.size(-1)
            else:
                dim_normalizer = 1.0
            # reduce to one number
            if reduction == "none":
                losses = losses.masked_fill(mask, float("nan"))
            else:
                losses = losses.masked_fill(mask, 0.0) / dim_normalizer
        return losses

    def reduce_loss(self, losses: torch.Tensor, lengths: torch.Tensor, reduction: str = None) -> torch.Tensor:
        """
        Reduce loss.

        Args:
            losses (torch.Tensor): total loss. first dimenion are samples, second timesteps
            lengths (torch.Tensor): total length
            reduction (str, optional): type of reduction. Defaults to ``self.reduction``.

        Returns:
            torch.Tensor: reduced loss
        """
        if reduction is None:
            reduction = self.reduction
        if reduction == "none":
            return losses  # return immediately, no checks
        elif reduction == "mean":
            loss = losses.sum() / lengths.sum()
        elif reduction == "sqrt-mean":
            loss = losses.sum() / lengths.sum()
            loss = loss.sqrt()
        else:
            raise ValueError(f"reduction {reduction} unknown")
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

    def loss(self, y_pred: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        return F.poisson_nll_loss(
            super().to_prediction(y_pred), target, log_input=True, full=False, eps=1e-6, reduction="none"
        )

    def to_prediction(self, out: Dict[str, torch.Tensor]):
        rate = torch.exp(super().to_prediction(out))
        return rate

    def to_quantiles(self, out: Dict[str, torch.Tensor], quantiles=None):
        if quantiles is None:
            if self.quantiles is None:
                quantiles = [0.5]
            else:
                quantiles = self.quantiles
        predictions = super().to_prediction(out)
        return torch.stack([torch.tensor(scipy.stats.poisson(predictions).ppf(q)) for q in quantiles], dim=-1)


class QuantileLoss(MultiHorizonMetric):
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calcualted as

    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """

    def __init__(
        self,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        **kwargs,
    ):
        """
        Quantile loss

        Args:
            quantiles: quantiles for metric
        """
        super().__init__(quantiles=quantiles, **kwargs)

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

    def loss(self, y_pred, target):
        y_pred = self.to_prediction(y_pred)
        loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
        return loss


class MAPE(MultiHorizonMetric):
    """
    Mean average percentage. Assumes ``y >= 0``.

    Defined as ``(y - y_pred).abs() / y.abs()``
    """

    def loss(self, y_pred, target):
        loss = (self.to_prediction(y_pred) - target).abs() / (target.abs() + 1e-8)
        return loss


class MAE(MultiHorizonMetric):
    """
    Mean average absolute error.

    Defined as ``(y_pred - target).abs()``
    """

    def loss(self, y_pred, target):
        loss = (self.to_prediction(y_pred) - target).abs()
        return loss


class CrossEntropy(MultiHorizonMetric):
    """
    Cross entropy loss for classification.
    """

    def loss(self, y_pred, target):

        loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), target.view(-1), reduction="none").view(
            -1, target.size(-1)
        )
        return loss


class RMSE(MultiHorizonMetric):
    """
    Root mean square error

    Defined as ``(y_pred - target)**2``
    """

    def __init__(self, reduction="sqrt-mean", **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def loss(self, y_pred: Dict[str, torch.Tensor], target):
        loss = torch.pow(self.to_prediction(y_pred) - target, 2)
        return loss


class MASE(MultiHorizonMetric):
    """
    Mean absolute scaled error

    Defined as ``(y_pred - target).abs() / all_targets[:, :-1] - all_targets[:, 1:]).mean(1)``.
    ``all_targets`` are here the concatenated encoder and decoder targets
    """

    def update(
        self,
        y_pred: Dict[str, torch.Tensor],
        target: Union[torch.Tensor, rnn.PackedSequence],
        encoder_target: Union[torch.Tensor, rnn.PackedSequence],
        encoder_lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Update metric that handles masking of values.

        Args:
            y_pred (Dict[str, torch.Tensor]): network output
            target (Union[torch.Tensor, rnn.PackedSequence]): actual values
            encoder_target (Union[torch.Tensor, rnn.PackedSequence]): historic actual values
            encoder_lengths (torch.Tensor): optional encoder lengths, not necessary if encoder_target
                is rnn.PackedSequence. Assumed encoder_target is torch.Tensor

        Returns:
            torch.Tensor: loss as a single number for backpropagation
        """
        target, lengths = unpack_sequence(target)
        if encoder_lengths is None:
            encoder_target, encoder_lengths = unpack_sequence(target)
        else:
            assert isinstance(encoder_target, torch.Tensor)
        assert not target.requires_grad

        # calculate loss with "none" reduction
        if target.ndim == 3:
            weight = target[..., 1]
            target = target[..., 0]
        else:
            weight = None

        scaling = self.calculate_scaling(target, lengths, encoder_target, encoder_lengths)
        losses = self.loss(y_pred, target, scaling)
        # weight samples
        if weight is not None:
            losses = losses * weight.unsqueeze(-1)

        self._update_losses_and_lengths(losses, lengths)

    def loss(self, y_pred, target, scaling):
        return (y_pred - target).abs() / scaling.unsqueeze(-1)

    def calculate_scaling(self, target, lengths, encoder_target, encoder_lengths):
        # calcualte mean(abs(diff(targets)))
        eps = 1e-6
        batch_size = target.size(0)
        total_lengths = lengths + encoder_lengths
        assert (total_lengths > 1).all(), "Need at least 2 target values to be able to calculate MASE"
        max_length = target.size(1) + encoder_target.size(1)
        if (total_lengths != max_length).any():  # if decoder or encoder targets have sequences of different lengths
            targets = torch.cat(
                [
                    encoder_target,
                    torch.zeros(batch_size, target.size(1), device=target.device, dtype=encoder_target.dtype),
                ],
                dim=1,
            )
            target_index = torch.arange(target.size(1), device=target.device, dtype=torch.long).unsqueeze(0).expand(
                batch_size, -1
            ) + encoder_lengths.unsqueeze(-1)
            targets.scatter_(dim=1, src=target, index=target_index)
        else:
            targets = torch.cat([encoder_target, target], dim=1)

        # take absolute difference
        diffs = (targets[:, :-1] - targets[:, 1:]).abs()

        # set last difference to 0
        not_maximum_length = total_lengths != max_length
        zero_correction_indices = total_lengths[not_maximum_length] - 1
        if len(zero_correction_indices) > 0:
            diffs[
                torch.arange(batch_size, dtype=torch.long, device=diffs.device)[not_maximum_length],
                zero_correction_indices,
            ] = 0.0

        # calculate mean over differences
        scaling = diffs.sum(1) / total_lengths + eps

        return scaling
