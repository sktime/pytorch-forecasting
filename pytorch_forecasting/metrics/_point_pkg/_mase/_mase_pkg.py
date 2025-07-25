"""
Package container for the Mean Absolute Scaled Error (MASE) metric for point forecasts.
"""

from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class MASE_pkg(_BasePtMetric):
    """
    Mean Average scaled Error (MASE) metric for point forecasts.
    """

    _tags = {"metric_type": "point", "info:metric_name": "MASE"}

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics import MASE

        return MASE
