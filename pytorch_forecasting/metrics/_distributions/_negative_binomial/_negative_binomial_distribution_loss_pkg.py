"""
Package container for the Negative Binomial distribution loss metric.
"""

from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


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
    def get_metric_cls(cls):
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
