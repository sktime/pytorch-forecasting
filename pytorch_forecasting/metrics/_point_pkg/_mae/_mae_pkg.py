"""
Package container for the Mean Absolute Error (MAE) metric for point forecasts.
"""

from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class MAE_pkg(_BasePtMetric):
    """
    Mean Average Error (MAE) metric for point forecasts.

    Defined as ``(y_pred - target).abs()``.
    """

    _tags = {
        "metric_type": "point",
        "requires:data_type": "point_forecast",
        "info:metric_name": "MAE",
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics import MAE

        return MAE
