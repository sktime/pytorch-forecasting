"""
Package container for the MQF2 distribution loss metric.
"""

from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class MQF2DistributionLoss_pkg(_BasePtMetric):
    """
    MQF2 distribution loss metric for distribution forecasts.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "mqf2",
        "info:metric_name": "MQF2DistributionLoss",
        "python_dependencies": ["cpflows"],
        "capability:quantile_generation": True,
        "requires:data_type": "mqf2_distribution_forecast",
        "clip_target": True,
        "data_loader_kwargs": {
            "target_normalizer": GroupNormalizer(
                groups=["agency", "sku"], center=False, transformation="log1p"
            )
        },
        "trainer_kwargs": dict(accelerator="cpu"),
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.distributions import MQF2DistributionLoss

        return MQF2DistributionLoss

    @classmethod
    def get_encoder(cls):
        """
        Returns a TorchNormalizer instance for rescaling parameters.
        """
        return TorchNormalizer()

    @classmethod
    def get_metric_test_params(cls):
        """
        Returns test parameters for the MQF2 distribution loss metric.
        """

        return [
            {
                "prediction_length": 10,
            },
        ]
