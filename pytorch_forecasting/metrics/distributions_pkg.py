"""
Package containers for all distribution-forecast metrics.
"""

import torch

from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.metrics.base._base_object import _BasePtMetric


class NormalDistributionLoss_pkg(_BasePtMetric):
    """
    Normal distribution loss metric for distribution forecasts.

    Defined as ``(y_pred - target)**2``.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "normal",
        "info:metric_name": "NormalDistributionLoss",
    }

    @classmethod
    def get_model_cls(cls):
        from pytorch_forecasting.metrics.distributions import NormalDistributionLoss

        return NormalDistributionLoss


class MultivariateNormalDistributionLoss_pkg(_BasePtMetric):
    """
    Multivariate normal distribution loss metric for distribution forecasts.

    Defined as ``(y_pred - target)**2``.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "multivariate_normal",
        "info:metric_name": "MultivariateNormalDistributionLoss",
    }

    @classmethod
    def get_model_cls(cls):
        from pytorch_forecasting.metrics.distributions import (
            MultivariateNormalDistributionLoss,
        )

        return MultivariateNormalDistributionLoss


class NegativeBinomialDistributionLoss_pkg(_BasePtMetric):
    """
    Negative binomial distribution loss metric for distribution forecasts.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "negative_binomial",
        "info:metric_name": "NegativeBinomialDistributionLoss",
    }

    @classmethod
    def get_model_cls(cls):
        from pytorch_forecasting.metrics.distributions import (
            NegativeBinomialDistributionLoss,
        )

        return NegativeBinomialDistributionLoss

    @classmethod
    def get_encoder(cls):
        """
        Returns a TorchNormalizer instance for rescaling parameters.
        """
        return TorchNormalizer(center=False)


class LogNormalDistributionLoss_pkg(_BasePtMetric):
    """
    Log-normal distribution loss metric for distribution forecasts.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "log_normal",
        "info:metric_name": "LogNormalDistributionLoss",
    }

    @classmethod
    def get_model_cls(cls):
        from pytorch_forecasting.metrics.distributions import LogNormalDistributionLoss

        return LogNormalDistributionLoss

    @classmethod
    def get_encoder(cls):
        """
        Returns a TorchNormalizer instance for rescaling parameters.
        """
        return TorchNormalizer(transformation="log")

    @classmethod
    def prepare_test_inputs(cls, test_case):
        """Prepare inputs for log normal distribution tests."""
        y_pred = test_case["y_pred"]
        y = test_case["y"]

        if isinstance(y, torch.nn.utils.rnn.PackedSequence):
            data, lengths = torch.nn.utils.rnn.pad_packed_sequence(y, batch_first=True)
            data = torch.where(data <= 0, torch.tensor(1e-4, device=data.device), data)

            y = torch.nn.utils.rnn.pack_padded_sequence(
                data, lengths, batch_first=True, enforce_sorted=False
            )

        return y_pred, y


class BetaDistributionLoss_pkg(_BasePtMetric):
    """
    Beta distribution loss metric for distribution forecasts.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "beta",
        "info:metric_name": "BetaDistributionLoss",
    }

    @classmethod
    def get_model_cls(cls):
        from pytorch_forecasting.metrics.distributions import BetaDistributionLoss

        return BetaDistributionLoss

    @classmethod
    def get_encoder(cls):
        """
        Returns a TorchNormalizer instance for rescaling parameters.
        """
        return TorchNormalizer(transformation="logit")
