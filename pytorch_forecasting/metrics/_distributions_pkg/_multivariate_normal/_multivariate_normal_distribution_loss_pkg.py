"""
Package container for multivariate normal distribution loss metric.
"""

from pytorch_forecasting.data.encoders import GroupNormalizer
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
        "requires:data_type": "multivariate_normal_distribution_forecast",
        "info:pred_type": ["distr"],
        "info:y_type": ["numeric"],
        "loss_ndim": 2,
    }

    @property
    def clip_target(self):
        return False

    @property
    def data_loader_kwargs(self):
        return {
            "target_normalizer": GroupNormalizer(
                groups=["agency", "sku"], transformation="log1p"
            )
        }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.distributions import (
            MultivariateNormalDistributionLoss,
        )

        return MultivariateNormalDistributionLoss

    @classmethod
    def _get_test_dataloaders(cls, params=None):
        """
        Returns test dataloaders configured for MultivariateNormalDistributionLoss.
        """
        kwargs = dict(target="agency")
        kwargs.update(cls.data_loader_kwargs)
        return super()._get_test_dataloaders_from(params, **kwargs)
