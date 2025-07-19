"""
Package container for multivariate normal distribution loss metric.
"""

from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


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
    def get_cls(cls):
        from pytorch_forecasting.metrics.distributions import (
            MultivariateNormalDistributionLoss,
        )

        return MultivariateNormalDistributionLoss
