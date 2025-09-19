"""
Package container for the Beta distribution loss metric.
"""

from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class BetaDistributionLoss_pkg(_BasePtMetric):
    """
    Beta distribution loss metric for distribution forecasts.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "beta",
        "info:metric_name": "BetaDistributionLoss",
        "requires:data_type": "beta_distribution_forecast",
        "info:pred_type": ["distr"],
        "info:y_type": ["numeric"],
        "loss_ndim": 2,
    }

    @property
    def clip_target(cls):
        return True

    @property
    def data_loader_kwargs(cls):
        return {
            "target_normalizer": GroupNormalizer(
                groups=["agency", "sku"], transformation="logit"
            )
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

    @classmethod
    def _get_test_dataloaders(cls, params=None):
        """
        Returns test dataloaders configured for BetaDistributionLoss.
        """
        kwargs = dict(target="agency")
        kwargs.update(cls.data_loader_kwargs)
        return super()._get_test_dataloaders_from(
            params, clip_target=cls.clip_target, **kwargs
        )
