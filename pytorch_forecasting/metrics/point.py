"""Point metrics for forecasting a single point per time step."""

import scipy.stats
import torch
import torch.nn.functional as F
from torch.nn.utils import rnn

from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric
from pytorch_forecasting.utils import unpack_sequence


class PoissonLoss(MultiHorizonMetric):
    """
    Poisson loss for count data.

    The loss will take the exponential of the network output before it is returned as prediction.
    Target normalizer should therefore have no "reverse" transformation, e.g.
    for the :py:class:`~data.timeseries.TimeSeriesDataSet` initialization, one could use:

    .. code-block:: python

        from pytorch_forecasting import TimeSeriesDataSet, EncoderNormalizer

        dataset = TimeSeriesDataSet(
            target_normalizer=EncoderNormalizer(transformation=dict(forward=torch.log1p))
        )

    Note that in this example, the data is log1p-transformed before normalized but not re-transformed.
    The PoissonLoss applies this "exp"-re-transformation on the network output after it has been de-normalized.
    The result is the model prediction.
    """  # noqa: E501

    def loss(
        self, y_pred: dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        return F.poisson_nll_loss(
            super().to_prediction(y_pred),
            target,
            log_input=True,
            full=False,
            eps=1e-6,
            reduction="none",
        )

    def to_prediction(self, out: dict[str, torch.Tensor]):
        rate = torch.exp(super().to_prediction(out))
        return rate

    def to_quantiles(self, out: dict[str, torch.Tensor], quantiles=None):
        if quantiles is None:
            if self.quantiles is None:
                quantiles = [0.5]
            else:
                quantiles = self.quantiles
        predictions = self.to_prediction(out)
        return (
            torch.stack(
                [
                    torch.tensor(
                        scipy.stats.poisson(predictions.detach().cpu().numpy()).ppf(q)
                    )
                    for q in quantiles
                ],
                dim=-1,
            )
            .type(predictions.dtype)
            .to(predictions.device)
        )


class SMAPE(MultiHorizonMetric):
    """
    Symmetric mean absolute percentage. Assumes ``y >= 0``.

    Defined as ``2*(y - y_pred).abs() / (y.abs() + y_pred.abs())``
    """

    def loss(self, y_pred, target):
        y_pred = self.to_prediction(y_pred)
        loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
        return loss


class MAPE(MultiHorizonMetric):
    """
    Mean absolute percentage. Assumes ``y >= 0``.

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
        loss = F.cross_entropy(
            y_pred.view(-1, y_pred.size(-1)), target.view(-1), reduction="none"
        ).view(-1, target.size(-1))
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
        return y_pred


class RMSE(MultiHorizonMetric):
    """
    Root mean square error

    Defined as ``(y_pred - target)**2``
    """

    def __init__(self, reduction="sqrt-mean", **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def loss(self, y_pred: dict[str, torch.Tensor], target):
        loss = torch.pow(self.to_prediction(y_pred) - target, 2)
        return loss


class MASE(MultiHorizonMetric):
    """
    Mean absolute scaled error

    Defined as ``(y_pred - target).abs() / all_targets[:, :-1] - all_targets[:, 1:]).mean(1)``.
    ``all_targets`` are here the concatenated encoder and decoder targets
    """  # noqa: E501

    def update(
        self,
        y_pred,
        target,
        encoder_target,
        encoder_lengths=None,
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
        """  # noqa: E501
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
            lengths = torch.full(
                (target.size(0),),
                fill_value=target.size(1),
                dtype=torch.long,
                device=target.device,
            )

        # determine lengths for encoder
        if encoder_lengths is None:
            encoder_target, encoder_lengths = unpack_sequence(encoder_target)
        else:
            assert isinstance(encoder_target, torch.Tensor)
        assert not target.requires_grad

        # calculate loss with "none" reduction
        scaling = self.calculate_scaling(
            target, lengths, encoder_target, encoder_lengths
        )
        losses = self.loss(y_pred, target, scaling)

        # weight samples
        if weight is not None:
            losses = losses * weight.unsqueeze(-1)

        self._update_losses_and_lengths(losses, lengths)

    def loss(self, y_pred, target, scaling):
        return (self.to_prediction(y_pred) - target).abs() / scaling.unsqueeze(-1)

    @staticmethod
    def calculate_scaling(target, lengths, encoder_target, encoder_lengths):
        # calcualte mean(abs(diff(targets)))
        eps = 1e-6
        batch_size = target.size(0)
        total_lengths = lengths + encoder_lengths
        assert (
            total_lengths > 1
        ).all(), "Need at least 2 target values to be able to calculate MASE"
        max_length = target.size(1) + encoder_target.size(1)
        if (
            total_lengths != max_length
        ).any():  # if decoder or encoder targets have sequences of different lengths
            targets = torch.cat(
                [
                    encoder_target,
                    torch.zeros(
                        batch_size,
                        target.size(1),
                        device=target.device,
                        dtype=encoder_target.dtype,
                    ),
                ],
                dim=1,
            )
            target_index = torch.arange(
                target.size(1), device=target.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1) + encoder_lengths.unsqueeze(-1)
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
                torch.arange(batch_size, dtype=torch.long, device=diffs.device)[
                    not_maximum_length
                ],
                zero_correction_indices,
            ] = 0.0

        # calculate mean over differences
        scaling = diffs.sum(1) / total_lengths + eps

        return scaling


class TweedieLoss(MultiHorizonMetric):
    """
    Tweedie loss.

    Tweedie regression with log-link. It might be useful, e.g., for modeling total
    loss in insurance, or for any target that might be tweedie-distributed.

    The loss will take the exponential of the network output before it is returned as prediction.
    Target normalizer should therefore have no "reverse" transformation, e.g.
    for the :py:class:`~data.timeseries.TimeSeriesDataSet` initialization, one could use:

    .. code-block:: python

        from pytorch_forecasting import TimeSeriesDataSet, EncoderNormalizer

        dataset = TimeSeriesDataSet(
            target_normalizer=EncoderNormalizer(transformation=dict(forward=torch.log1p))
        )

    Note that in this example, the data is log1p-transformed before normalized but not re-transformed.
    The TweedieLoss applies this "exp"-re-transformation on the network output after it has been de-normalized.
    The result is the model prediction.
    """  # noqa: E501

    def __init__(self, reduction="mean", p: float = 1.5, **kwargs):
        """
        Args:
            p (float, optional): tweedie variance power which is greater equal
                1.0 and smaller 2.0. Close to ``2`` shifts to
                Gamma distribution and close to ``1`` shifts to Poisson distribution.
                Defaults to 1.5.
            reduction (str, optional): How to reduce the loss. Defaults to "mean".
        """
        super().__init__(reduction=reduction, **kwargs)
        assert 1 <= p < 2, "p must be in range [1, 2]"
        self.p = p

    def to_prediction(self, out: dict[str, torch.Tensor]):
        rate = torch.exp(super().to_prediction(out))
        return rate

    def loss(self, y_pred, y_true):
        y_pred = super().to_prediction(y_pred)
        a = y_true * torch.exp(y_pred * (1 - self.p)) / (1 - self.p)
        b = torch.exp(y_pred * (2 - self.p)) / (2 - self.p)
        loss = -a + b
        return loss
