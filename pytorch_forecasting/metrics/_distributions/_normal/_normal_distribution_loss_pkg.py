"""
Package container for the Normal distribution loss metric.
"""

from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


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
