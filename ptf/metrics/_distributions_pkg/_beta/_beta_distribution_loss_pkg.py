"""
Package container for the Beta distribution loss metric.
"""

from ptf.data import TorchNormalizer
from ptf.metrics.base_metrics._base_object import _BasePtMetric


class BetaDistributionLoss_pkg(_BasePtMetric):
    """
    Beta distribution loss metric for distribution forecasts.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "beta",
        "info:metric_name": "BetaDistributionLoss",
        "requires:data_type": "beta_distribution_forecast",
    }

    @classmethod
    def get_cls(cls):
        from ptf.metrics.distributions import BetaDistributionLoss

        return BetaDistributionLoss

    @classmethod
    def get_encoder(cls):
        """
        Returns a TorchNormalizer instance for rescaling parameters.
        """
        return TorchNormalizer(transformation="logit")
