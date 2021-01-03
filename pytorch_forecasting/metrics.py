"""
Implementation of metrics for (mulit-horizon) timeseries forecasting.
"""
from typing import Dict, List, Tuple, Union
import warnings

from pandas.core.algorithms import isin
from pytorch_lightning.metrics import Metric as LightningMetric
import scipy.stats
from sklearn.base import BaseEstimator
import torch
from torch import distributions
import torch.nn.functional as F
from torch.nn.utils import rnn

from pytorch_forecasting.utils import create_mask, unpack_sequence, unsqueeze_like


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
            if self.quantiles is not None:
                idx = self.quantiles.index(0.5)
                y_pred = y_pred[..., idx]
            else:
                assert y_pred.size(-1) == 1, "Prediction should only have one extra dimension"
                y_pred = y_pred[..., 0]
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


class MultiLoss(LightningMetric):
    """
    Metric that can be used with muliple metrics.
    """

    def __init__(self, metrics: List[LightningMetric], weights: List[float] = None):
        """
        Args:
            metrics (List[LightningMetric], optional): list of metrics to combine.
            weights (List[float], optional): list of weights / multipliers for weights. Defaults to 1.0 for all metrics.
        """
        assert len(metrics) > 0, "at least one metric has to be specified"
        if weights is None:
            weights = [1.0 for _ in metrics]
        assert len(weights) == len(metrics), "Number of weights has to match number of metrics"

        self.metrics = metrics
        self.weights = weights

        super().__init__()

    def __repr__(self):
        name = (
            f"{self.__class__.__name__}("
            + ", ".join([f"{w:.3g} * {repr(m)}" if w != 1.0 else repr(m) for w, m in zip(self.weights, self.metrics)])
            + ")"
        )
        return name

    def __iter__(self):
        """
        Iterate over metrics.
        """
        return iter(self.metrics)

    def __len__(self) -> int:
        """
        Number of metrics.

        Returns:
            int: number of metrics
        """
        return len(self.metrics)

    def update(self, y_pred: torch.Tensor, y_actual: torch.Tensor):
        """
        Update composite metric

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        for idx, metric in enumerate(self.metrics):
            metric.update(y_pred[idx], (y_actual[0][idx], y_actual[1]))

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
        return [metric.to_prediction(y_pred[idx]) for idx, metric in enumerate(self.metrics)]

    def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Will use first metric in ``metrics`` attribute to calculate result.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: prediction quantiles
        """
        return [metric.to_quantiles(y_pred[idx]) for idx, metric in enumerate(self.metrics)]

    def __getitem__(self, idx: int):
        """
        Return metric.

        Args:
            idx (int): metric index
        """
        return self.metrics[idx]

    def __getattr__(self, name: str):
        """
        Return dynamically attributes.

        Return attributes if defined in this class. If not, create dynamically attributes based on
        attributes of underlying metrics that are lists. Create functions if necessary.
        Arguments to functions are distributed to the functions if they are lists and their length
        matches the number of metrics. Otherwise, they are directly passed to each callable of the
        metrics

        Args:
            name (str): name of attribute

        Returns:
            attributes of this class or list of attributes of underlying class
        """
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            attribute_exists = all([hasattr(metric, name) for metric in self.metrics])
            if attribute_exists:
                # check if to return callable or not and return function if yes
                if callable(getattr(self.metrics[0], name)):
                    n = len(self.metrics)

                    def func(*args, **kwargs):
                        # if arg/kwarg is list and of length metric, then apply each part to a metric. otherwise
                        # pass it directly to all metrics
                        results = []
                        for idx, m in enumerate(self.metrics):
                            new_args = [
                                arg[idx]
                                if isinstance(arg, (list, tuple))
                                and not isinstance(arg, rnn.PackedSequence)
                                and len(arg) == n
                                else arg
                                for arg in args
                            ]
                            new_kwargs = {
                                key: val[idx]
                                if isinstance(val, list) and not isinstance(val, rnn.PackedSequence) and len(val) == n
                                else val
                                for key, val in kwargs.items()
                            }
                            results.append(getattr(m, name)(*new_args, **new_kwargs))
                        return results

                    return func
                else:
                    # else return list of attributes
                    return [getattr(metric, name) for metric in self.metrics]
            else:  # attribute does not exist for all metrics
                raise e


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
        # extract target and weight
        if isinstance(y_actual, (tuple, list)) and not isinstance(y_actual, rnn.PackedSequence):
            target, weight = y_actual
        else:
            target = y_actual
            weight = None

        # handle rnn sequence as target
        if isinstance(target, rnn.PackedSequence):
            target, lengths = rnn.pad_packed_sequence(target, batch_first=True)
            # batch sizes reside on the CPU by default -> we need to bring them to GPU
            lengths = lengths.to(target.device)

            # calculate mask for time steps
            length_mask = create_mask(target.size(1), lengths, inverse=True)

            # modify weight
            if weight is None:
                weight = length_mask
            else:
                weight = weight * length_mask

        if weight is None:
            y_mean = target.mean(0)
            y_pred_mean = y_pred.mean(0)
        else:

            # calculate weighted sums
            y_mean = (target * unsqueeze_like(weight, y_pred)).sum(0) / weight.sum(0)

            y_pred_sum = (y_pred * unsqueeze_like(weight, y_pred)).sum(0)
            y_pred_mean = y_pred_sum / unsqueeze_like(weight.sum(0), y_pred_sum)

        # update metric. unsqueeze first batch dimension (as batches are collapsed)
        self.metric.update(y_pred_mean.unsqueeze(0), y_mean.unsqueeze(0))

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
        # unpack weight
        if isinstance(target, (list, tuple)) and not isinstance(target, rnn.PackedSequence):
            target, weight = target
        else:
            weight = None

        # unpack target
        if isinstance(target, rnn.PackedSequence):
            target, lengths = unpack_sequence(target)
        else:
            lengths = torch.full((target.size(0),), fill_value=target.size(1), dtype=torch.long, device=target.device)

        losses = self.loss(y_pred, target)
        # weight samples
        if weight is not None:
            losses = losses * unsqueeze_like(weight, losses)
        self._update_losses_and_lengths(losses, lengths)

    def _update_losses_and_lengths(self, losses: torch.Tensor, lengths: torch.Tensor):
        losses = self.mask_losses(losses, lengths)
        if self.reduction == "none":
            if self.losses.ndim == 0:
                self.losses = losses
                self.lengths = lengths
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

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Returns best label

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        return y_pred.argmax(dim=-1)


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
        target: Tuple[Union[torch.Tensor, rnn.PackedSequence], torch.Tensor],
        encoder_target: Union[torch.Tensor, rnn.PackedSequence],
        encoder_lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Update metric that handles masking of values.

        Args:
            y_pred (Dict[str, torch.Tensor]): network output
            target (Tuple[Union[torch.Tensor, rnn.PackedSequence], torch.Tensor]): tuple of actual values and weights
            encoder_target (Union[torch.Tensor, rnn.PackedSequence]): historic actual values
            encoder_lengths (torch.Tensor): optional encoder lengths, not necessary if encoder_target
                is rnn.PackedSequence. Assumed encoder_target is torch.Tensor

        Returns:
            torch.Tensor: loss as a single number for backpropagation
        """
        # unpack weight
        if isinstance(target, (list, tuple)):
            weight = target[1]
            target = target[0]
        else:
            weight = None

        # unpack target
        if isinstance(target, rnn.PackedSequence):
            target, lengths = unpack_sequence(target)
        else:
            lengths = torch.full((target.size(0),), fill_value=target.size(1), dtype=torch.long, device=target.device)

        # determine lengths for encoder
        if encoder_lengths is None:
            encoder_target, encoder_lengths = unpack_sequence(target)
        else:
            assert isinstance(encoder_target, torch.Tensor)
        assert not target.requires_grad

        # calculate loss with "none" reduction
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


class DistributionLoss(MultiHorizonMetric):
    """
    DistributionLoss base class.

    Class should be inherited for all distribution losses, i.e. if a network predicts
    the parameters of a probability distribution, DistributionLoss can be used to
    score those parameters and calculate loss for given true values.

    Define two class attributes in a child class:

    Attributes:
        distribution_class (distributions.Distribution): torch probability distribution
        distribution_arguments (List[str]): list of parameter names for the distribution

    Further, implement the methods :py:meth:`~map_x_to_distribution` and :py:meth:`~rescale_parameters`.

    """

    distribution_class: distributions.Distribution
    distribution_arguments: List[str]

    def __init__(
        self, name: str = None, quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98], reduction="mean"
    ):
        """
        Initialize metric

        Args:
            name (str): metric name. Defaults to class name.
            quantiles (List[float], optional): quantiles for probability range.
                Defaults to [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98].
            reduction (str, optional): Reduction, "none", "mean" or "sqrt-mean". Defaults to "mean".
        """
        super().__init__(name=name, quantiles=quantiles, reduction=reduction)

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Distribution:
        """
        Map the a tensor of parameters to a probability distribution.

        Args:
            x (torch.Tensor): parameters for probability distribution. Last dimension will index the parameters

        Returns:
            distributions.Distribution: torch probability distribution as defined in the
                class attribute ``distribution_class``
        """
        raise NotImplementedError("implement this method")

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative likelihood

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        distribution = self.map_x_to_distribution(y_pred)
        loss = -distribution.log_prob(y_actual)
        return loss

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        """
        Rescale normalized parameters into the scale required for the distribution.

        Args:
            parameters (torch.Tensor): normalized parameters (indexed by last dimension)
            target_scale (torch.Tensor): scale of parameters (n_batch_samples x (center, scale))
            encoder (BaseEstimator): original encoder that normalized the target in the first place

        Returns:
            torch.Tensor: parameters in real/not normalized space
        """
        return parameters

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network (with ``output_transformation = None``)

        Returns:
            torch.Tensor: mean prediction
        """
        return y_pred.mean(-1)

    def sample(self, y_pred, n_samples: int) -> torch.Tensor:
        """
        Sample from distribution.

        Args:
            y_pred: prediction output of network (shape batch_size x n_timesteps x n_paramters)
            n_samples (int): number of samples to draw

        Returns:
            torch.Tensor: tensor with samples  (shape batch_size x n_timesteps x n_samples)
        """
        dist = self.map_x_to_distribution(y_pred)
        samples = dist.sample((n_samples,))
        if samples.ndim == 3:
            samples = samples.permute(1, 2, 0)
        elif samples.ndim == 2:
            samples = samples.transpose(0, 1)
        return samples

    def to_quantiles(self, y_pred: torch.Tensor, quantiles: List[float] = None) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network (with ``output_transformation = None``)
            quantiles (List[float], optional): quantiles for probability range. Defaults to quantiles as
                as defined in the class initialization.

        Returns:
            torch.Tensor: prediction quantiles (last dimension)
        """
        if quantiles is None:
            quantiles = self.quantiles

        samples = y_pred.size(-1)
        quantiles = torch.stack(
            [torch.kthvalue(y_pred, int(samples * q), dim=-1)[0] if samples > 1 else y_pred[..., 0] for q in quantiles],
            dim=-1,
        )
        return quantiles


class NormalDistributionLoss(DistributionLoss):
    """
    Normal distribution loss.

    Requirements for original target normalizer:
        * not normalized in log space (use :py:class:`~LogNormalDistributionLoss`)
        * not coerced to be positive
    """

    distribution_class = distributions.Normal
    distribution_arguments = ["loc", "scale"]

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        return self.distribution_class(loc=x[..., 0], scale=x[..., 1])

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        assert encoder.transformation not in ["log", "log1p"], "Use LogNormalDistributionLoss for log scaled data"
        assert encoder.transformation not in [
            "softplus",
            "relu",
        ], "Cannot use NormalDistributionLoss for positive data"
        assert encoder.transformation not in ["logit"], "Cannot use bound transformation such as 'logit'"
        loc = encoder(dict(prediction=parameters[..., 0], target_scale=target_scale))
        scale = F.softplus(parameters[..., 1]) * target_scale[..., 1].unsqueeze(1)
        return torch.stack([loc, scale], dim=-1)


class NegativeBinomialDistributionLoss(DistributionLoss):
    """
    Negative binomial loss, e.g. for count data.

    Requirements for original target normalizer:
        * not centered normalization (only rescaled)
    """

    distribution_class = distributions.NegativeBinomial
    distribution_arguments = ["mean", "shape"]

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.NegativeBinomial:
        mean = x[..., 0]
        shape = x[..., 1]
        r = 1.0 / shape
        p = mean / (mean + r)
        return self.distribution_class(total_count=r, probs=p)

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        assert not encoder.center, "NegativeBinomialDistributionLoss is not compatible with `center=True` normalization"
        assert encoder.transformation not in ["logit"], "Cannot use bound transformation such as 'logit'"
        if encoder.transformation in ["log", "log1p"]:
            mean = torch.exp(parameters[..., 0] * target_scale[..., 1].unsqueeze(-1))
            shape = (
                F.softplus(torch.exp(parameters[..., 1]))
                / torch.exp(target_scale[..., 1].unsqueeze(-1)).sqrt()  # todo: is this correct?
            )
        else:
            mean = F.softplus(parameters[..., 0]) * target_scale[..., 1].unsqueeze(-1)
            shape = F.softplus(parameters[..., 1]) / target_scale[..., 1].unsqueeze(-1).sqrt()
        return torch.stack([mean, shape], dim=-1)


class LogNormalDistributionLoss(DistributionLoss):
    """
    Log-normal loss.

    Requirements for original target normalizer:
        * normalized target in log space
    """

    distribution_class = distributions.LogNormal
    distribution_arguments = ["loc", "scale"]

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.LogNormal:
        return self.distribution_class(loc=x[..., 0], scale=x[..., 1])

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        assert isinstance(encoder.transformation, str) and encoder.transformation in [
            "log",
            "log1p",
        ], f"Log distribution requires log scaling but found `transformation={encoder.transform}`"

        assert encoder.transformation not in ["logit"], "Cannot use bound transformation such as 'logit'"

        scale = F.softplus(parameters[..., 1]) * target_scale[..., 1].unsqueeze(-1)
        loc = parameters[..., 0] * target_scale[..., 1].unsqueeze(-1) + target_scale[..., 0].unsqueeze(-1)

        return torch.stack([loc, scale], dim=-1)


class BetaDistributionLoss(DistributionLoss):
    """
    Beta distribution loss for unit interval data.

    Requirements for original target normalizer:
        * logit transformation
    """

    distribution_class = distributions.Beta
    distribution_arguments = ["mean", "shape"]
    eps = 1e-4

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Beta:
        mean = x[..., 0]
        shape = x[..., 1]
        return self.distribution_class(concentration0=(1 - mean) * shape, concentration1=mean * shape)

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative likelihood

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        distribution = self.map_x_to_distribution(y_pred)
        # clip y_actual to avoid infinite losses
        loss = -distribution.log_prob(y_actual.clip(self.eps, 1 - self.eps))
        return loss

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        assert encoder.transformation in ["logit"], "Beta distribution is only compatible with logit transformation"
        assert encoder.center, "Beta distribution requires normalizer to center data"

        scaled_mean = encoder(dict(prediction=parameters[..., 0], target_scale=target_scale))
        # need to first transform target scale standard deviation in logit space to real space
        # we assume a normal distribution in logit space (we used a logit transform and a standard scaler)
        # and know that the variance of the beta distribution is limited by `scaled_mean * (1 - scaled_mean)`
        scaled_mean = scaled_mean * (1 - 2 * self.eps) + self.eps  # ensure that mean is not exactly 0 or 1
        mean_derivative = scaled_mean * (1 - scaled_mean)

        # we can approximate variance as
        # torch.pow(torch.tanh(target_scale[..., 1].unsqueeze(1) * torch.sqrt(mean_derivative)), 2) * mean_derivative
        # shape is (positive) parameter * mean_derivative / var
        shape_scaler = (
            torch.pow(torch.tanh(target_scale[..., 1].unsqueeze(1) * torch.sqrt(mean_derivative)), 2) + self.eps
        )
        scaled_shape = F.softplus(parameters[..., 1]) / shape_scaler
        return torch.stack([scaled_mean, scaled_shape], dim=-1)
