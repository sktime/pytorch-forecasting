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
        "compatible_pred_types": ["distr"],
        "compatible_y_types": ["numeric"],
        "expected_loss_ndim": 2,
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.distributions import (
            MultivariateNormalDistributionLoss,
        )

        return MultivariateNormalDistributionLoss

    @classmethod
    def _get_test_dataloaders_from(cls, params=None):
        """
        Returns test dataloaders configured for MultivariateNormalDistributionLoss.
        """
        from pytorch_forecasting.tests._data_scenarios import data_with_covariates, make_dataloaders

        if params is None:
            params = {}
        data_loader_kwargs = cls._tags.get("data_loader_kwargs", {}).copy()
        data_loader_kwargs.update(params.get("data_loader_kwargs", {}))

        data = data_with_covariates()
        dataloaders = make_dataloaders(data, **data_loader_kwargs)
        return dataloaders