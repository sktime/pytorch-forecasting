"""Metrics that allow the parametric forecast of parameters of uni- and multivariate distributions."""  # noqa: E501

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
import torch
from torch import distributions, nn
import torch.nn.functional as F

from pytorch_forecasting.data.encoders import TorchNormalizer, softplus_inv
from pytorch_forecasting.metrics.base_metrics import (
    DistributionLoss,
    MultivariateDistributionLoss,
)


class NormalDistributionLoss(DistributionLoss):
    """
    Normal distribution loss.
    """

    distribution_class = distributions.Normal
    distribution_arguments = ["loc", "scale"]

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        distr = self.distribution_class(loc=x[..., 2], scale=x[..., 3])
        scaler = distributions.AffineTransform(loc=x[..., 0], scale=x[..., 1])
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            return distributions.TransformedDistribution(
                distr,
                [
                    scaler,
                    TorchNormalizer.get_transform(self._transformation)[
                        "inverse_torch"
                    ],
                ],
            )

    def rescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        self._transformation = encoder.transformation
        loc = parameters[..., 0]
        scale = F.softplus(parameters[..., 1])
        return torch.concat(
            [
                target_scale.unsqueeze(1).expand(-1, loc.size(1), -1),
                loc.unsqueeze(-1),
                scale.unsqueeze(-1),
            ],
            dim=-1,
        )


class MultivariateNormalDistributionLoss(MultivariateDistributionLoss):
    """
    Multivariate low-rank normal distribution loss.

    Use this loss to make out of a DeepAR model a DeepVAR network.
    """

    distribution_class = distributions.LowRankMultivariateNormal

    def __init__(
        self,
        name: str = None,
        quantiles: Optional[list[float]] = None,
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
        """  # noqa: E501
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        super().__init__(name=name, quantiles=quantiles, reduction=reduction)
        self.rank = rank
        self.sigma_minimum = sigma_minimum
        self.sigma_init = sigma_init
        self.distribution_arguments = list(range(2 + rank))

        # determine bias
        self._diag_bias: float = (
            softplus_inv(torch.tensor(self.sigma_init) ** 2).item()
            if self.sigma_init > 0.0
            else 0.0
        )
        # determine normalizer to bring unscaled diagonal close to 1.0
        self._cov_factor_scale: float = np.sqrt(self.rank)

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        assert x.device.type != "mps", (
            "MPS accelerator has a bug"
            " https://github.com/pytorch/pytorch/issues/98074, use cpu or gpu"
        )
        x = x.permute(1, 0, 2)
        distr = self.distribution_class(
            loc=x[..., 2],
            cov_factor=x[..., 4:],
            cov_diag=x[..., 3],
        )
        scaler = distributions.AffineTransform(
            loc=x[0, :, 0], scale=x[0, :, 1], event_dim=1
        )
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            return distributions.TransformedDistribution(
                distr,
                [
                    scaler,
                    TorchNormalizer.get_transform(self._transformation)[
                        "inverse_torch"
                    ],
                ],
            )

    def rescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        self._transformation = encoder.transformation

        # scale
        loc = parameters[..., 0].unsqueeze(-1)
        scale = (
            F.softplus(parameters[..., 1].unsqueeze(-1) + self._diag_bias)
            + self.sigma_minimum**2
        )

        cov_factor = parameters[..., 2:] / self._cov_factor_scale
        return torch.concat(
            [
                target_scale.unsqueeze(1).expand(-1, loc.size(1), -1),
                loc,
                scale,
                cov_factor,
            ],
            dim=-1,
        )


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
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        assert not encoder.center, (
            "NegativeBinomialDistributionLoss is not"
            " compatible with `center=True` normalization"
        )
        assert encoder.transformation not in [
            "logit",
            "log",
        ], "Cannot use bound transformation such as 'logit'"
        if encoder.transformation in ["log1p"]:
            mean = torch.exp(parameters[..., 0] * target_scale[..., 1].unsqueeze(-1))
            shape = (
                F.softplus(torch.exp(parameters[..., 1]))
                / torch.exp(
                    target_scale[..., 1].unsqueeze(-1)
                ).sqrt()  # todo: is this correct?
            )
        else:
            mean = F.softplus(parameters[..., 0]) * target_scale[..., 1].unsqueeze(-1)
            shape = (
                F.softplus(parameters[..., 1])
                / target_scale[..., 1].unsqueeze(-1).sqrt()
            )
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
        """  # noqa: E501
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
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        assert isinstance(encoder.transformation, str) and encoder.transformation in [
            "log",
            "log1p",
        ], (
            "Log distribution requires log scaling but found"
            f" `transformation={encoder.transform}`"
        )

        assert encoder.transformation not in [
            "logit"
        ], "Cannot use bound transformation such as 'logit'"

        scale = F.softplus(parameters[..., 1]) * target_scale[..., 1].unsqueeze(-1)
        loc = parameters[..., 0] * target_scale[..., 1].unsqueeze(-1) + target_scale[
            ..., 0
        ].unsqueeze(-1)

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
        return self.distribution_class(
            concentration0=(1 - mean) * shape, concentration1=mean * shape
        )

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
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        assert encoder.transformation in [
            "logit"
        ], "Beta distribution is only compatible with logit transformation"
        assert encoder.center, "Beta distribution requires normalizer to center data"

        scaled_mean = encoder(
            dict(prediction=parameters[..., 0], target_scale=target_scale)
        )
        # need to first transform target scale standard deviation in
        # logit space to real space
        # we assume a normal distribution in logit space
        # (we used a logit transform and a standard scaler)
        # and know that the variance of the beta distribution is
        # limited by `scaled_mean * (1 - scaled_mean)`
        scaled_mean = (
            scaled_mean * (1 - 2 * self.eps) + self.eps
        )  # ensure that mean is not exactly 0 or 1
        mean_derivative = scaled_mean * (1 - scaled_mean)

        # we can approximate variance as
        # torch.pow(torch.tanh(target_scale[..., 1].unsqueeze(1) * torch.sqrt(mean_derivative)), 2) * mean_derivative # noqa: E501
        # shape is (positive) parameter * mean_derivative / var
        shape_scaler = (
            torch.pow(
                torch.tanh(
                    target_scale[..., 1].unsqueeze(1) * torch.sqrt(mean_derivative)
                ),
                2,
            )
            + self.eps
        )
        scaled_shape = F.softplus(parameters[..., 1]) / shape_scaler
        return torch.stack([scaled_mean, scaled_shape], dim=-1)


class MQF2DistributionLoss(DistributionLoss):
    """Multivariate quantile loss based on the article
    `Multivariate Quantile Function Forecaster <http://arxiv.org/abs/2202.11316>`_.

    Requires install of additional library:
    ``pip install pytorch-forecasting[mqf2]``
    """

    eps = 1e-4

    def __init__(
        self,
        prediction_length: int,
        quantiles: Optional[list[float]] = None,
        hidden_size: Optional[int] = 4,
        es_num_samples: int = 50,
        beta: float = 1.0,
        icnn_hidden_size: int = 20,
        icnn_num_layers: int = 2,
        estimate_logdet: bool = False,
    ) -> None:
        """
        Args:
            prediction_length (int): maximum prediction length.
            quantiles (List[float], optional): default quantiles to output.
                Defaults to [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98].
            hidden_size (int, optional): hidden size per prediction length. Defaults to 4.
            es_num_samples (int, optional): Number of samples to calculate energy score.
                If None, maximum liklihood is used as opposed to energy score for optimization.
                Defaults to 50.
            beta (float, optional): between 0 and 1.0 to control how scale sensitive metric is (1=fully sensitive).
                Defaults to 1.0.
            icnn_hidden_size (int, optional): hidden size of distribution estimating network. Defaults to 20.
            icnn_num_layers (int, optional): number of hidden layers in distribution estimating network. Defaults to 2.
            estimate_logdet (bool, optional): if to estimate log determinant. Defaults to False.
        """  # noqa: E501
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        super().__init__(quantiles=quantiles)

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
        scaler = distributions.AffineTransform(loc=loc, scale=scale)
        if self._transformation is None:
            return self.transformed_distribution_class(distr, [scaler])
        else:
            return self.transformed_distribution_class(
                distr,
                [
                    scaler,
                    TorchNormalizer.get_transform(self._transformation)[
                        "inverse_torch"
                    ],
                ],
            )

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
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        self._transformation = encoder.transformation
        return torch.concat(
            [parameters.reshape(parameters.size(0), -1), target_scale], dim=-1
        )

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
            torch.Tensor: prediction quantiles (last dimension)
        """  # noqa: E501
        if quantiles is None:
            quantiles = self.quantiles
        distribution = self.map_x_to_distribution(y_pred)
        alpha = (
            torch.as_tensor(quantiles, device=y_pred.device)[:, None]
            .repeat(y_pred.size(0), 1)
            .expand(-1, self.prediction_length)
        )
        hidden_state = distribution.base_dist.hidden_state.repeat_interleave(
            len(quantiles), dim=0
        )
        result = distribution.quantile(
            alpha, hidden_state=hidden_state
        )  # (batch_size * quantiles x prediction_length)

        # reshape
        result = result.reshape(-1, len(quantiles), self.prediction_length).transpose(
            1, 2
        )  # (batch_size, prediction_length, quantile_size)

        return result


class ImplicitQuantileNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.quantile_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, input_size),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.PReLU(),
            nn.Linear(input_size, 1),
        )
        self.register_buffer("cos_multipliers", torch.arange(0, hidden_size) * torch.pi)

    def forward(self, x: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
        # embed quantiles
        cos_emb_tau = torch.cos(
            quantiles[:, None] * self.cos_multipliers[None]
        )  # n_quantiles x hidden_size
        # modulates input depending on quantile
        cos_emb_tau = self.quantile_layer(cos_emb_tau)  # n_quantiles x input_size

        emb_inputs = x.unsqueeze(-2) * (
            1.0 + cos_emb_tau
        )  # ... x n_quantiles x input_size
        emb_outputs = self.output_layer(emb_inputs).squeeze(-1)  # ... x n_quantiles
        return emb_outputs


class ImplicitQuantileNetworkDistributionLoss(DistributionLoss):
    """Implicit Quantile Network Distribution Loss.

    Based on `Probabilistic Time Series Forecasting with Implicit Quantile Networks
    <https://arxiv.org/pdf/2107.03743.pdf>`_.
    A network is used to directly map network outputs to a quantile.
    """

    def __init__(
        self,
        quantiles: Optional[list[float]] = None,
        input_size: Optional[int] = 16,
        hidden_size: Optional[int] = 32,
        n_loss_samples: Optional[int] = 64,
    ) -> None:
        """
        Args:
            prediction_length (int): maximum prediction length.
            quantiles (List[float], optional): default quantiles to output.
                Defaults to [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98].
            input_size (int, optional): input size per prediction length. Defaults to 16.
            hidden_size (int, optional): hidden size per prediction length. Defaults to 64.
            n_loss_samples (int, optional): number of quantiles to sample to calculate loss.
        """  # noqa: E501
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        super().__init__(quantiles=quantiles)
        self.quantile_network = ImplicitQuantileNetwork(
            input_size=input_size, hidden_size=hidden_size
        )
        self.distribution_arguments = list(range(int(input_size)))
        self.n_loss_samples = n_loss_samples

    def sample(self, y_pred, n_samples: int) -> torch.Tensor:
        eps = 1e-3
        # for a couple of random quantiles
        # (excl. 0 and 1 as they would lead to infinities)
        quantiles = torch.rand(size=(n_samples,), device=y_pred.device).clamp(
            eps, 1 - eps
        )
        # make prediction
        samples = self.to_quantiles(y_pred, quantiles=quantiles)
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
        eps = 1e-3
        # for a couple of random quantiles
        # (excl. 0 and 1 as they would lead to infinities)
        quantiles = torch.rand(size=(self.n_loss_samples,), device=y_pred.device).clamp(
            eps, 1 - eps
        )
        # make prediction
        pred_quantiles = self.to_quantiles(y_pred, quantiles=quantiles)
        # and calculate quantile loss
        errors = y_actual[..., None] - pred_quantiles
        loss = 2 * torch.fmax(
            quantiles[None] * errors, (quantiles[None] - 1) * errors
        ).mean(dim=-1)
        return loss

    def rescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        self._transformation = encoder.transformation
        return torch.concat(
            [parameters, target_scale.unsqueeze(1).expand(-1, parameters.size(1), -1)],
            dim=-1,
        )

    def to_prediction(self, y_pred: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        if n_samples is None:
            return self.to_quantiles(y_pred, quantiles=[0.5]).squeeze(-1)
        else:
            # for a couple of random quantiles
            # (excl. 0 and 1 as they would lead to infinities) make prediction
            return self.sample(y_pred, n_samples=n_samples).mean(-1)

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
            torch.Tensor: prediction quantiles (last dimension)
        """  # noqa: E501
        if quantiles is None:
            quantiles = self.quantiles
        quantiles = torch.as_tensor(quantiles, device=y_pred.device)

        # extract parameters
        x = y_pred[..., :-2]
        loc = y_pred[..., -2][..., None]
        scale = y_pred[..., -1][..., None]

        # predict quantiles
        if y_pred.requires_grad:
            predictions = self.quantile_network(x, quantiles)
        else:
            with torch.no_grad():
                predictions = self.quantile_network(x, quantiles)
        # rescale output
        predictions = loc + predictions * scale
        # transform output if required
        if self._transformation is not None:
            transform = TorchNormalizer.get_transform(self._transformation)["reverse"]
            predictions = transform(predictions)

        return predictions
