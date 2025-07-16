"""
Package container for Symmetric Mean Absolute Percentage Error (SMAPE) metric for point
forecasts.
"""

from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class SMAPE_pkg(_BasePtMetric):
    """
    Symmetric mean absolute percentage error metric for point forecasts.

    Defined as ``2*(y - y_pred).abs() / (y.abs() + y_pred.abs())``.
    Assumes ``y >= 0``.
    """

    _tags = {"metric_type": "point", "info:metric_name": "SMAPE"}

    @classmethod
    def get_metric_cls(cls):
        from pytorch_forecasting.metrics.point import SMAPE

        return SMAPE
