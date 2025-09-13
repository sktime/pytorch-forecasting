"""
Package container for the Negative Binomial distribution loss metric.
"""

from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class NegativeBinomialDistributionLoss_pkg(_BasePtMetric):
    """
    Negative binomial distribution loss metric for distribution forecasts.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "negative_binomial",
        "info:metric_name": "NegativeBinomialDistributionLoss",
        "requires:data_type": "negative_binomial_distribution_forecast",
        "info:pred_type": ["distr"],
        "info:y_type": ["numeric"],
        "loss_ndim": 2,
    }

    clip_target = False
    data_loader_kwargs = {
        "target_normalizer": GroupNormalizer(groups=["agency", "sku"], center=False)
    }

    @classmethod
    def get_cls(cls):
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

    @classmethod
    def _get_test_dataloaders(cls, params=None):
        """
        Returns test dataloaders configured for NegativeBinomialDistributionLoss.
        """
        kwargs = dict(target="agency")
        kwargs.update(cls.data_loader_kwargs)
        return super()._get_test_dataloaders_from(
            params, clip_target=cls.clip_target, **kwargs
        )
