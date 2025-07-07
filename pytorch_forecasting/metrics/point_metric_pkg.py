"""
Package containers for all point-forecast metrics.
"""

from pytorch_forecasting.metrics.base._base_object import _BasePtMetric


class MAE_pkg(_BasePtMetric):
    """
    Mean Average Error (MAE) metric for point forecasts.

    Defined as ``(y_pred - target).abs()``.
    """

    _tags = {"metric_type": "point", "info:metric_name": "MAE"}

    @classmethod
    def _get_metric_cls(cls):
        from pytorch_forecasting.metrics import MAE

        return MAE
