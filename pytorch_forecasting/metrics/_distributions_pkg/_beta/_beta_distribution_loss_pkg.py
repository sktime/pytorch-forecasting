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
        "requires:data_type": "beta_distribution_forecast",
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
    def get_default_params(cls):
        from pytorch_forecasting.data.encoders import GroupNormalizer

        return {
            "clip_target": True,
            "data_loader_kwargs": {
                "target_normalizer": GroupNormalizer(
                    groups=["agency", "sku"], transformation="logit"
                )
            },
        }
