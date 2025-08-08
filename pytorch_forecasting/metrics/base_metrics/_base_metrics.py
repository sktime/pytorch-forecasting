"""
Base classes for metrics - only for inheritance.
"""

import inspect
from typing import Any, Callable, Optional
import warnings

from sklearn.base import BaseEstimator
import torch
from torch import distributions
from torch.nn.utils import rnn
from torchmetrics import Metric as LightningMetric

from pytorch_forecasting.utils import create_mask, unpack_sequence, unsqueeze_like


class Metric(LightningMetric):
    """
    Base metric class that has basic functions that can handle predicting quantiles and operate in log space.
    See the `Lightning documentation <https://pytorch-lightning.readthedocs.io/en/latest/metrics.html>`_
    for details of how to implement a new metric

    Other metrics should inherit from this base class
    """  # noqa: E501

    full_state_update = False
    higher_is_better = False
    is_differentiable = True

    def __init__(
        self,
        name: str = None,
        quantiles: list[float] = None,
        reduction="mean",
        **kwargs,
    ):
        """
        Initialize metric

        Args:
            name (str): metric name. Defaults to class name.
            quantiles (List[float], optional): quantiles for probability range. Defaults to None.
            reduction (str, optional): Reduction, "none", "mean" or "sqrt-mean". Defaults to "mean".
        """  # noqa: E501
        self.quantiles = quantiles
        self.reduction = reduction
        if name is None:
            name = self.__class__.__name__
        self.name = name
        super().__init__(**kwargs)

    def update(self, y_pred: torch.Tensor, y_actual: torch.Tensor):
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

    def rescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        """
        Rescale normalized parameters into the scale required for the output.

        Args:
            parameters (torch.Tensor): normalized parameters (indexed by last dimension)
            target_scale (torch.Tensor): scale of parameters (n_batch_samples x (center, scale))
            encoder (BaseEstimator): original encoder that normalized the target in the first place

        Returns:
            torch.Tensor: parameters in real/not normalized space
        """  # noqa: E501
        return encoder(dict(prediction=parameters, target_scale=target_scale))

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
                assert (
                    y_pred.size(-1) == 1
                ), "Prediction should only have one extra dimension"
                y_pred = y_pred[..., 0]
            else:
                y_pred = y_pred.mean(-1)
        return y_pred

    def to_quantiles(
        self, y_pred: torch.Tensor, quantiles: list[float] = None
    ) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network
            quantiles (List[float], optional): quantiles for probability range. Defaults to quantiles as
                as defined in the class initialization.

        Returns:
            torch.Tensor: prediction quantiles
        """  # noqa: E501
        if quantiles is None:
            quantiles = self.quantiles

        if y_pred.ndim == 2:
            return y_pred.unsqueeze(-1)
        elif y_pred.ndim == 3:
            if y_pred.size(2) > 1:  # single dimension means all quantiles are the same
                assert quantiles is not None, "quantiles are not defined"
                y_pred = torch.quantile(
                    y_pred, torch.tensor(quantiles, device=y_pred.device), dim=2
                ).permute(1, 2, 0)
            return y_pred
        else:
            raise ValueError(
                f"prediction has 1 or more than 3 dimensions: {y_pred.ndim}"
            )

    def __add__(self, metric: LightningMetric):
        composite_metric = CompositeMetric(metrics=[self])
        new_metric = composite_metric + metric
        return new_metric

    def __mul__(self, multiplier: float):
        new_metric = CompositeMetric(metrics=[self], weights=[multiplier])
        return new_metric

    def extra_repr(self) -> str:
        forbidden_attributes = ["name", "reduction"]
        attributes = list(inspect.signature(self.__class__).parameters.keys())
        return ", ".join(
            [
                f"{name}={repr(getattr(self, name))}"
                for name in attributes
                if hasattr(self, name) and name not in forbidden_attributes
            ]
        )

    __rmul__ = __mul__


class TorchMetricWrapper(Metric):
    """
    Wrap a torchmetric to work with PyTorch Forecasting.

    Does not support weighting of errors and only supports metrics for point predictions.
    """  # noqa: E501

    def __init__(self, torchmetric: LightningMetric, reduction: str = None, **kwargs):
        """
        Args:
            torchmetric (LightningMetric): Torchmetric to wrap.
            reduction (str, optional): use reduction with torchmetric directly. Defaults to None.
        """  # noqa: E501
        super().__init__(**kwargs)
        if reduction is not None:
            raise ValueError("use reduction with torchmetric directly")
        self.torchmetric = torchmetric

    def _sync_dist(self, dist_sync_fn=None, process_group=None) -> None:
        # No syncing required here. syncing will be done in metric_a and metric_b
        pass

    def _wrap_compute(self, compute: Callable) -> Callable:
        return compute

    def reset(self) -> None:
        self.torchmetric.reset()

    def persistent(self, mode: bool = False) -> None:
        self.torchmetric.persistent(mode=mode)

    def _convert(
        self, y_pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # unpack target into target and weights
        if isinstance(target, (list, tuple)) and not isinstance(
            target, rnn.PackedSequence
        ):
            target, weight = target
            if weight is not None:
                raise NotImplementedError(
                    "Weighting is not supported for pure torchmetrics - "
                    "implement a custom version or use pytorch-forecasting metrics"
                )

        # convert to point prediction - limits applications of class
        y_pred = self.to_prediction(y_pred)

        # unpack target if it is PackedSequence
        if isinstance(target, rnn.PackedSequence):
            target, lengths = unpack_sequence(target)
            # create mask for different lengths
            length_mask = create_mask(target.size(1), lengths, inverse=True)
            target = target.masked_select(length_mask)
            y_pred = y_pred.masked_select(length_mask)

        y_pred = y_pred.flatten()
        target = target.flatten()
        return y_pred, target

    def update(
        self, y_pred: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        # flatten target and prediction
        y_pred_flattened, target_flattened = self._convert(y_pred, target)

        # update metric
        self.torchmetric.update(y_pred_flattened, target_flattened, **kwargs)

    def forward(self, y_pred, target, **kwargs):
        # need this explicitly to avoid backpropagation
        # errors because of sketchy caching
        y_pred_flattened, target_flattened = self._convert(y_pred, target)
        return self.torchmetric.forward(y_pred_flattened, target_flattened, **kwargs)

    def compute(self):
        res = self.torchmetric.compute()
        return res

    def __repr__(self):
        return f"WrappedTorchmetric({repr(self.torchmetric)})"


def convert_torchmetric_to_pytorch_forecasting_metric(
    metric: LightningMetric,
) -> Metric:
    """
    If necessary, convert a torchmetric to a PyTorch Forecasting metric that
    works with PyTorch Forecasting models.

    Args:
        metric (LightningMetric): metric to (potentially) convert

    Returns:
        Metric: PyTorch Forecasting metric
    """
    if not isinstance(metric, (Metric, MultiLoss, CompositeMetric)):
        return TorchMetricWrapper(metric)
    else:
        return metric


class MultiLoss(LightningMetric):
    """
    Metric that can be used with muliple metrics.
    """

    full_state_update = False
    higher_is_better = False
    is_differentiable = True

    def __init__(self, metrics: list[LightningMetric], weights: list[float] = None):
        """
        Args:
            metrics (List[LightningMetric], optional): list of metrics to combine.
            weights (List[float], optional): list of weights / multipliers for weights. Defaults to 1.0 for all metrics.
        """  # noqa: E501
        assert len(metrics) > 0, "at least one metric has to be specified"
        if weights is None:
            weights = [1.0 for _ in metrics]
        assert len(weights) == len(
            metrics
        ), "Number of weights has to match number of metrics"

        self.metrics = [
            convert_torchmetric_to_pytorch_forecasting_metric(m) for m in metrics
        ]
        self.weights = weights

        super().__init__()

    def __repr__(self):
        name = (
            f"{self.__class__.__name__}("
            + ", ".join(
                [
                    f"{w:.3g} * {repr(m)}" if w != 1.0 else repr(m)
                    for w, m in zip(self.weights, self.metrics)
                ]
            )
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

    def update(self, y_pred: torch.Tensor, y_actual: torch.Tensor, **kwargs) -> None:
        """
        Update composite metric

        Args:
            y_pred: network output
            y_actual: actual values
            **kwargs: arguments to update function
        """
        for idx, metric in enumerate(self.metrics):
            try:
                metric.update(
                    y_pred[idx],
                    (y_actual[0][idx], y_actual[1]),
                    **{
                        name: value[idx] if isinstance(value, (list, tuple)) else value
                        for name, value in kwargs.items()
                    },
                )
            except TypeError:  # silently update without kwargs if not supported
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

    @torch.jit.unused
    def forward(self, y_pred: torch.Tensor, y_actual: torch.Tensor, **kwargs):
        """
        Calculate composite metric

        Args:
            y_pred: network output
            y_actual: actual values
            **kwargs: arguments to update function

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        results = []
        for idx, metric in enumerate(self.metrics):
            try:
                res = metric(
                    y_pred[idx],
                    (y_actual[0][idx], y_actual[1]),
                    **{
                        name: value[idx] if isinstance(value, (list, tuple)) else value
                        for name, value in kwargs.items()
                    },
                )
            except TypeError:  # silently update without kwargs if not supported
                res = metric(y_pred[idx], (y_actual[0][idx], y_actual[1]))
            results.append(res * self.weights[idx])

        if len(results) == 1:
            results = results[0]
        else:
            results = torch.stack(results, dim=0).sum(0)
        return results

    def _wrap_compute(self, compute: Callable) -> Callable:
        return compute

    def _sync_dist(
        self,
        dist_sync_fn: Optional[Callable] = None,
        process_group: Optional[Any] = None,
    ) -> None:
        # No syncing required here. syncing will be done in metrics
        pass

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()

    def persistent(self, mode: bool = False) -> None:
        for metric in self.metrics:
            metric.persistent(mode=mode)

    def to_prediction(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Will use first metric in ``metrics`` attribute to calculate result.

        Args:
            y_pred: prediction output of network
            **kwargs: arguments for metrics

        Returns:
            torch.Tensor: point prediction
        """
        result = []
        for idx, metric in enumerate(self.metrics):
            try:
                result.append(metric.to_prediction(y_pred[idx], **kwargs))
            except TypeError:
                result.append(metric.to_prediction(y_pred[idx]))
        return result

    def to_quantiles(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Will use first metric in ``metrics`` attribute to calculate result.

        Args:
            y_pred: prediction output of network
            **kwargs: parameters to each metric's ``to_quantiles()`` method

        Returns:
            torch.Tensor: prediction quantiles
        """
        result = []
        for idx, metric in enumerate(self.metrics):
            try:
                result.append(metric.to_quantiles(y_pred[idx], **kwargs))
            except TypeError:
                result.append(metric.to_quantiles(y_pred[idx]))
        return result

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
        """  # noqa: E501
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            attribute_exists = all(hasattr(metric, name) for metric in self.metrics)
            if attribute_exists:
                # check if to return callable or not and return function if yes
                if callable(getattr(self.metrics[0], name)):
                    n = len(self.metrics)

                    def func(*args, **kwargs):
                        # if arg/kwarg is list and of length metric,
                        # then apply each part to a metric. otherwise
                        # pass it directly to all metrics
                        results = []
                        for idx, m in enumerate(self.metrics):
                            new_args = [
                                (
                                    arg[idx]
                                    if isinstance(arg, (list, tuple))
                                    and not isinstance(arg, rnn.PackedSequence)
                                    and len(arg) == n
                                    else arg
                                )
                                for arg in args
                            ]
                            new_kwargs = {
                                key: (
                                    val[idx]
                                    if isinstance(val, list)
                                    and not isinstance(val, rnn.PackedSequence)
                                    and len(val) == n
                                    else val
                                )
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
    """  # noqa: E501

    full_state_update = False
    higher_is_better = False
    is_differentiable = True

    def __init__(
        self,
        metrics: Optional[list[LightningMetric]] = None,
        weights: Optional[list[float]] = None,
    ):
        """
        Args:
            metrics (List[LightningMetric], optional): list of metrics to combine. Defaults to None.
            weights (List[float], optional): list of weights / multipliers for weights. Defaults to 1.0 for all metrics.
        """  # noqa: E501
        self.metrics = metrics
        self.weights = weights

        if metrics is None:
            metrics = []
        if weights is None:
            weights = [1.0 for _ in metrics]
        assert len(weights) == len(
            metrics
        ), "Number of weights has to match number of metrics"

        self._metrics = list(metrics)
        self._weights = list(weights)

        super().__init__()

    def __repr__(self):
        name = " + ".join(
            [
                f"{w:.3g} * {repr(m)}" if w != 1.0 else repr(m)
                for w, m in zip(self._weights, self._metrics)
            ]
        )
        return name

    def update(self, y_pred: torch.Tensor, y_actual: torch.Tensor, **kwargs):
        """
        Update composite metric

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        for metric in self._metrics:
            try:
                metric.update(y_pred, y_actual, **kwargs)
            except TypeError:
                metric.update(y_pred, y_actual)

    def compute(self) -> torch.Tensor:
        """
        Get metric

        Returns:
            torch.Tensor: metric
        """
        results = []
        for weight, metric in zip(self._weights, self._metrics):
            results.append(metric.compute() * weight)

        if len(results) == 1:
            results = results[0]
        else:
            results = torch.stack(results, dim=0).sum(0)
        return results

    @torch.jit.unused
    def forward(self, y_pred: torch.Tensor, y_actual: torch.Tensor, **kwargs):
        """
        Calculate composite metric

        Args:
            y_pred: network output
            y_actual: actual values
            **kwargs: arguments to update function

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        results = []
        for weight, metric in zip(self._weights, self._metrics):
            try:
                results.append(metric(y_pred, y_actual, **kwargs) * weight)
            except TypeError:
                results.append(metric(y_pred, y_actual) * weight)

        if len(results) == 1:
            results = results[0]
        else:
            results = torch.stack(results, dim=0).sum(0)
        return results

    def _wrap_compute(self, compute: Callable) -> Callable:
        return compute

    def _sync_dist(
        self,
        dist_sync_fn: Optional[Callable] = None,
        process_group: Optional[Any] = None,
    ) -> None:
        # No syncing required here. syncing will be done in metrics
        pass

    def reset(self) -> None:
        for metric in self._metrics:
            metric.reset()

    def persistent(self, mode: bool = False) -> None:
        for metric in self._metrics:
            metric.persistent(mode=mode)

    def to_prediction(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Will use first metric in ``metrics`` attribute to calculate result.

        Args:
            y_pred: prediction output of network
            **kwargs: parameters to first metric `to_prediction` method

        Returns:
            torch.Tensor: point prediction
        """
        return self._metrics[0].to_prediction(y_pred, **kwargs)

    def to_quantiles(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Will use first metric in ``metrics`` attribute to calculate result.

        Args:
            y_pred: prediction output of network
            **kwargs: parameters to first metric's ``to_quantiles()`` method

        Returns:
            torch.Tensor: prediction quantiles
        """
        return self._metrics[0].to_quantiles(y_pred, **kwargs)

    def __add__(self, metric: LightningMetric):
        if isinstance(metric, self.__class__):
            self._metrics.extend(metric._metrics)
            self._weights.extend(metric._weights)
        else:
            self._metrics.append(metric)
            self._weights.append(1.0)

        return self

    def __mul__(self, multiplier: float):
        self._weights = [w * multiplier for w in self._weights]
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

    def update(
        self, y_pred: torch.Tensor, y_actual: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Calculate composite metric

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        y_pred_mean, y_mean = self._calculate_mean(y_pred, y_actual)
        # update metric. unsqueeze first batch dimension (as batches are collapsed)
        self.metric.update(y_pred_mean, y_mean, **kwargs)

    @staticmethod
    def _calculate_mean(
        y_pred: torch.Tensor, y_actual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # extract target and weight
        if isinstance(y_actual, (tuple, list)) and not isinstance(
            y_actual, rnn.PackedSequence
        ):
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
        return y_pred_mean.unsqueeze(0), y_mean.unsqueeze(0)

    def compute(self):
        return self.metric.compute()

    @torch.jit.unused
    def forward(self, y_pred: torch.Tensor, y_actual: torch.Tensor, **kwargs):
        """
        Calculate composite metric

        Args:
            y_pred: network output
            y_actual: actual values
            **kwargs: arguments to update function

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        y_pred_mean, y_mean = self._calculate_mean(y_pred, y_actual)
        return self.metric(y_pred_mean, y_mean, **kwargs)

    def _wrap_compute(self, compute: Callable) -> Callable:
        return compute

    def _sync_dist(
        self,
        dist_sync_fn: Optional[Callable] = None,
        process_group: Optional[Any] = None,
    ) -> None:
        # No syncing required here. syncing will be done in metrics
        pass

    def reset(self) -> None:
        self.metrics.reset()

    def persistent(self, mode: bool = False) -> None:
        self.metric.persistent(mode=mode)


class MultiHorizonMetric(Metric):
    """
    Abstract class for defining metric for a multihorizon forecast
    """

    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        super().__init__(reduction=reduction, **kwargs)
        if reduction == "none":
            default_losses = default_lengths = []
            dist_reduce_fx = "cat"
        else:
            default_losses = 0.0
            default_lengths = 0
            dist_reduce_fx = "sum"

        self.add_state(
            "losses",
            default=torch.tensor(default_losses, dtype=torch.float),
            dist_reduce_fx=dist_reduce_fx,
        )
        self.add_state(
            "lengths",
            default=torch.tensor(default_lengths, dtype=torch.long),
            dist_reduce_fx=dist_reduce_fx,
        )

    def loss(
        self, y_pred: dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate loss without reduction. Override in derived classes

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: loss/metric as a single number for backpropagation
        """
        raise NotImplementedError()

    def update(self, y_pred, target):
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
        if isinstance(target, (list, tuple)) and not isinstance(
            target, rnn.PackedSequence
        ):
            target, weight = target
        else:
            weight = None

        # unpack target
        if isinstance(target, rnn.PackedSequence):
            target, lengths = unpack_sequence(target)
        else:
            lengths = torch.full(
                (target.size(0),),
                fill_value=target.size(1),
                dtype=torch.long,
                device=target.device,
            )

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
                losses = torch.tensor(1e9, device=losses.device)
                warnings.warn("Loss is not finite. Resetting it to 1e9")
            self.losses = self.losses + losses
            self.lengths = self.lengths + lengths.sum()

    def compute(self):
        loss = self.reduce_loss(self.losses, lengths=self.lengths)
        return loss

    def mask_losses(
        self, losses: torch.Tensor, lengths: torch.Tensor, reduction: str = None
    ) -> torch.Tensor:
        """
        Mask losses.

        Args:
            losses (torch.Tensor): total loss. first dimenion are samples, second timesteps
            lengths (torch.Tensor): total length
            reduction (str, optional): type of reduction. Defaults to ``self.reduction``.

        Returns:
            torch.Tensor: masked losses
        """  # noqa: E501
        if reduction is None:
            reduction = self.reduction
        if losses.ndim > 0:
            # mask loss
            mask = torch.arange(losses.size(1), device=losses.device).unsqueeze(
                0
            ) >= lengths.unsqueeze(-1)
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

    def reduce_loss(
        self, losses: torch.Tensor, lengths: torch.Tensor, reduction: str = None
    ) -> torch.Tensor:
        """
        Reduce loss.

        Args:
            losses (torch.Tensor): total loss. first dimenion are samples, second timesteps
            lengths (torch.Tensor): total length
            reduction (str, optional): type of reduction. Defaults to ``self.reduction``.

        Returns:
            torch.Tensor: reduced loss
        """  # noqa: E501
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
        assert torch.isfinite(loss), (
            "Loss should not be infinite - i.e."
            " something went wrong (e.g. input is not in log space)"
        )
        return loss


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
    """  # noqa: E501

    distribution_class: distributions.Distribution
    distribution_arguments: list[str]

    def __init__(
        self,
        name: str = None,
        quantiles: Optional[list[float]] = None,
        reduction="mean",
    ):
        """
        Initialize metric

        Args:
            name (str): metric name. Defaults to class name.
            quantiles (List[float], optional): quantiles for probability range.
                Defaults to [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98].
            reduction (str, optional): Reduction, "none", "mean" or "sqrt-mean". Defaults to "mean".
        """  # noqa: E501
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        super().__init__(name=name, quantiles=quantiles, reduction=reduction)

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Distribution:
        """
        Map the a tensor of parameters to a probability distribution.

        Args:
            x (torch.Tensor): parameters for probability distribution. Last dimension will index the parameters

        Returns:
            distributions.Distribution: torch probability distribution as defined in the
                class attribute ``distribution_class``
        """  # noqa: E501
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

    def to_prediction(self, y_pred: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network
            n_samples (int): number of samples to draw
        Returns:
            torch.Tensor: mean prediction
        """
        distribution = self.map_x_to_distribution(y_pred)
        try:
            return distribution.mean
        except NotImplementedError:
            return self.sample(y_pred, n_samples=n_samples).mean(-1)

    def sample(self, y_pred, n_samples: int) -> torch.Tensor:
        """
        Sample from distribution.

        Args:
            y_pred: prediction output of network (shape batch_size x n_timesteps x n_paramters)
            n_samples (int): number of samples to draw

        Returns:
            torch.Tensor: tensor with samples  (shape batch_size x n_timesteps x n_samples)
        """  # noqa: E501
        dist = self.map_x_to_distribution(y_pred)
        samples = dist.sample((n_samples,))
        if samples.ndim == 3:
            samples = samples.permute(1, 2, 0)
        elif samples.ndim == 2:
            samples = samples.transpose(0, 1)
        return samples

    def to_quantiles(
        self, y_pred: torch.Tensor, quantiles: list[float] = None, n_samples: int = 100
    ) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network
            quantiles (List[float], optional): quantiles for probability range. Defaults to quantiles as
                as defined in the class initialization.
            n_samples (int): number of samples to draw for quantiles. Defaults to 100.

        Returns:
            torch.Tensor: prediction quantiles (last dimension)
        """  # noqa: E501
        if quantiles is None:
            quantiles = self.quantiles
        try:
            distribution = self.map_x_to_distribution(y_pred)
            quantiles = distribution.icdf(
                torch.tensor(quantiles, device=y_pred.device)[:, None, None]
            ).permute(1, 2, 0)
        except NotImplementedError:  # resort to derive quantiles empirically
            samples = torch.sort(self.sample(y_pred, n_samples), -1).values
            quantiles = torch.quantile(
                samples, torch.tensor(quantiles, device=samples.device), dim=2
            ).permute(1, 2, 0)
        return quantiles


class MultivariateDistributionLoss(DistributionLoss):
    """Base class for multivariate distribution losses.

    Class should be inherited for all multivariate distribution losses, i.e. if a batch of values
    is predicted in one go and the batch dimension is not independent, but the time dimension still
    remains independent.
    """  # noqa: E501

    def sample(self, y_pred, n_samples: int) -> torch.Tensor:
        """
        Sample from distribution.

        Args:
            y_pred: prediction output of network (shape batch_size x n_timesteps x n_paramters)
            n_samples (int): number of samples to draw

        Returns:
            torch.Tensor: tensor with samples  (shape batch_size x n_timesteps x n_samples)
        """  # noqa: E501
        dist = self.map_x_to_distribution(y_pred)
        samples = dist.sample(
            (n_samples,)
        ).permute(
            2, 1, 0
        )  # returned as (n_samples, n_timesteps, batch_size), so reshape to (batch_size, n_timesteps, n_samples) # noqa: E501
        return samples

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
        # calculate one number and scale with batch size
        loss = -distribution.log_prob(y_actual.transpose(0, 1)).sum() * y_actual.size(0)
        return loss
