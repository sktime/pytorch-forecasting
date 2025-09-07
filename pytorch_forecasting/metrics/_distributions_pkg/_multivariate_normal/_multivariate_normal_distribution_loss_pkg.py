"""
Package container for multivariate normal distribution loss metric.
"""

from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric
from pytorch_forecasting.data.encoders import GroupNormalizer

class MultivariateNormalDistributionLoss_pkg(_BasePtMetric):
    """
    Multivariate normal distribution loss metric for distribution forecasts.

    Defined as ``(y_pred - target)**2``.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "multivariate_normal",
        "info:metric_name": "MultivariateNormalDistributionLoss",
        "requires:data_type": "multivariate_normal_distribution_forecast",
        "data_loader_kwargs": {
            "target_normalizer": GroupNormalizer(
                groups=["agency", "sku"], transformation="log1p"
            )
        },
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.distributions import (
            MultivariateNormalDistributionLoss,
        )

        return MultivariateNormalDistributionLoss