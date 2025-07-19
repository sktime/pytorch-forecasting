"""
Package container for the Beta distribution loss metric.
"""

from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


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
    def get_cls(cls):
        from pytorch_forecasting.metrics.distributions import BetaDistributionLoss

        return BetaDistributionLoss

    @classmethod
    def get_encoder(cls):
        """
        Returns a TorchNormalizer instance for rescaling parameters.
        """
        return TorchNormalizer(transformation="logit")
