"""
Package containers for all distribution-forecast metrics.
"""

from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.metrics.base._base_object import _BasePtMetric


class NormalDistributionLoss_pkg(_BasePtMetric):
    """
    Normal distribution loss metric for distribution forecasts.

    Defined as ``(y_pred - target)**2``.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "normal",
        "info:metric_name": "NormalDistributionLoss",
    }

    @classmethod
    def get_model_cls(cls):
        from pytorch_forecasting.metrics.distributions import NormalDistributionLoss

        return NormalDistributionLoss

    @classmethod
    def prepare_test_inputs(cls, test_case):
        """
        Perform rescaling on input data for normal distribution loss.
        """

        y_pred_raw = test_case["y_pred"]
        y_raw = test_case["y"]

        metric_cls = cls.get_model_cls()

        y_pred = metric_cls().rescale_parameters(
            parameters=y_pred_raw,
            target_scale=test_case["x"]["target_scale"],
            encoder=TorchNormalizer(),
        )

        return y_pred, y_raw
