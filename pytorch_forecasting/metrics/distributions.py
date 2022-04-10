"""Metrics that allow the parametric forecast of parameters of uni- and multivariate distributions."""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
import torch
from torch import distributions
import torch.nn.functional as F

from pytorch_forecasting.metrics.base_metrics import DistributionLoss, MultivariateDistributionLoss


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


class MultivariateNormalDistributionLoss(MultivariateDistributionLoss):
    """
    Multivariate low-rank normal distribution loss.

    Use this loss to make out of a DeepAR model a DeepVAR network.

    Requirements for original target normalizer:
        * not normalized in log space (use :py:class:`~LogNormalDistributionLoss`)
        * not coerced to be positive
    """

    distribution_class = distributions.LowRankMultivariateNormal

    def __init__(
        self,
        name: str = None,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        reduction: str = "mean",
        rank: int = 10,
        sigma_init: float = 1.0,
        sigma_minimum: float = 1e-3,
    ):
        """
        Initialize metric

        Args:
            name (str): metric name. Defaults to class name.
            quantiles (List[float], optional): quantiles for probability range.
                Defaults to [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98].
            reduction (str, optional): Reduction, "none", "mean" or "sqrt-mean". Defaults to "mean".
            rank (int): rank of low-rank approximation for covariance matrix. Defaults to 10.
            sigma_init (float, optional): default value for diagonal covariance. Defaults to 1.0.
            sigma_minimum (float, optional): minimum value for diagonal covariance. Defaults to 1e-3.
        """
        super().__init__(name=name, quantiles=quantiles, reduction=reduction)
        self.rank = rank
        self.sigma_minimum = sigma_minimum
        self.sigma_init = sigma_init
        self.distribution_arguments = list(range(2 + rank))

        # determine bias
        self._diag_bias: float = self.inv_softplus(self.sigma_init**2) if self.sigma_init > 0.0 else 0.0

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        x = x.permute(1, 0, 2)
        return self.distribution_class(
            loc=x[..., 0],
            cov_factor=x[..., 2:],
            cov_diag=x[..., 1],
        )

    @staticmethod
    def validate_encoder(encoder: BaseEstimator):
        assert encoder.transformation not in [
            "log",
            "log1p",
        ], "Use MultivariateLogNormalDistributionLoss for log scaled data"  # todo: implement
        assert encoder.transformation not in [
            "softplus",
            "relu",
        ], "Cannot use NormalDistributionLoss for positive data"
        assert encoder.transformation not in ["logit"], "Cannot use bound transformation such as 'logit'"

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        self.validate_encoder(encoder)

        # scale
        loc = encoder(dict(prediction=parameters[..., 0], target_scale=target_scale)).unsqueeze(-1)
        scale = (
            F.softplus(parameters[..., 1].unsqueeze(-1) + self._diag_bias) + self.sigma_minimum**2
        ) * target_scale[..., 1, None, None] ** 2

        cov_factor = parameters[..., 2:] * target_scale[..., 1, None, None]
        return torch.concat([loc, scale, cov_factor], dim=-1)

    def inv_softplus(self, y):
        if y < 20.0:
            return np.log(np.exp(y) - 1.0)
        else:
            return y


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

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction. In the case of this distribution prediction we
        need to derive the mean (as a point prediction) from the distribution parameters

        Args:
            y_pred: prediction output of network
            in this case the two parameters for the negative binomial

        Returns:
            torch.Tensor: mean prediction
        """
        return y_pred[..., 0]


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


class MQF2DistributionLoss(DistributionLoss):
    """Multivariate quantile loss."""

    eps = 1e-4

    def __init__(
        self,
        prediction_length: int,
        hidden_size: int = 4,
        threshold_input: float = 100.0,
        es_num_samples: int = 50,
        beta: float = 1.0,
        icnn_hidden_size: int = 20,
        icnn_num_layers: int = 2,
        estimate_logdet: bool = False,
    ) -> None:
        super().__init__()

        from cpflows.flows import ActNorm
        from cpflows.icnn import PICNN

        from pytorch_forecasting.metrics._mqf2_utils import (
            DeepConvexNet,
            MQF2Distribution,
            SequentialNet,
            TransformedMQF2Distribution,
        )

        self.distribution_class = MQF2Distribution
        self.transformed_distribution_class = TransformedMQF2Distribution
        self.distribution_arguments = list(range(int(hidden_size)))
        self.prediction_length = prediction_length
        self.threshold_input = threshold_input
        self.es_num_samples = es_num_samples
        self.beta = beta

        # define picnn
        convexnet = PICNN(
            dim=prediction_length,
            dimh=icnn_hidden_size,
            dimc=hidden_size * prediction_length,
            num_hidden_layers=icnn_num_layers,
            symm_act_first=True,
        )
        deepconvexnet = DeepConvexNet(
            convexnet,
            prediction_length,
            is_energy_score=self.is_energy_score,
            estimate_logdet=estimate_logdet,
        )

        if self.is_energy_score:
            networks = [deepconvexnet]
        else:
            networks = [
                ActNorm(prediction_length),
                deepconvexnet,
                ActNorm(prediction_length),
            ]

        self.picnn = SequentialNet(networks)

    @property
    def is_energy_score(self) -> bool:
        return self.es_num_samples is not None

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Distribution:
        distr = self.distribution_class(
            picnn=self.picnn,
            hidden_state=x[..., :-2],
            prediction_length=self.prediction_length,
            is_energy_score=self.is_energy_score,
            es_num_samples=self.es_num_samples,
            beta=self.beta,
        )
        # rescale
        loc = x[..., -2][:, None]
        scale = x[..., -1][:, None]
        return self.transformed_distribution_class(distr, [distributions.AffineTransform(loc=loc, scale=scale)])

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
        if self.is_energy_score:
            loss = distribution.energy_score(y_actual)
        else:
            loss = -distribution.log_prob(y_actual)
        return loss.reshape(-1, 1)

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        return torch.concat([parameters.reshape(parameters.size(0), -1), target_scale], dim=-1)

    def to_quantiles(self, y_pred: torch.Tensor, quantiles: List[float] = None) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network
            quantiles (List[float], optional): quantiles for probability range. Defaults to quantiles as
                as defined in the class initialization.

        Returns:
            torch.Tensor: prediction quantiles (last dimension)
        """
        if quantiles is None:
            quantiles = self.quantiles
        distribution = self.map_x_to_distribution(y_pred)
        alpha = (
            torch.as_tensor(quantiles, device=y_pred.device)[:, None]
            .repeat(y_pred.size(0), 1)
            .expand(-1, self.prediction_length)
        )
        hidden_state = distribution.base_dist.hidden_state.repeat_interleave(len(quantiles), dim=0)
        result = distribution.quantile(alpha, hidden_state=hidden_state)  # (batch_size * quantiles x prediction_length)

        # reshape
        result = result.reshape(-1, len(quantiles), self.prediction_length).transpose(
            1, 2
        )  # (batch_size, prediction_length, quantile_size)

        return result
