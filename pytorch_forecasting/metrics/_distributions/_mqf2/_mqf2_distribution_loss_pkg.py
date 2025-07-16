"""
Package container for the MQF2 distribution loss metric.
"""

from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class MQF2DistributionLoss_pkg(_BasePtMetric):
    """
    MQF2 distribution loss metric for distribution forecasts.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "mqf2",
        "info:metric_name": "MQF2DistributionLoss",
        "requires_soft_dependency": ["cpflows"],
        "capability:quantile_generation": True,
    }

    @classmethod
    def get_metric_cls(cls):
        from pytorch_forecasting.metrics.distributions import MQF2DistributionLoss

        return MQF2DistributionLoss

    @classmethod
    def get_encoder(cls):
        """
        Returns a TorchNormalizer instance for rescaling parameters.
        """
        return TorchNormalizer()

    @classmethod
    def prepare_test_inputs(cls, test_case):
        """Prepare inputs for MQF2 distribution tests."""
        y_pred = test_case["y_pred"]
        y = test_case["y"]
        return y_pred, y

    @classmethod
    def get_test_params(cls):
        """
        Returns test parameters for the MQF2 distribution loss metric.
        """

        return {
            "prediction_length": 10,
        }
