"""
Package container for Root Mean Square Error (RMSE) metric for point forecasts.
"""

from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class RMSE_pkg(_BasePtMetric):
    """
    Root mean square error metric for point forecasts.

    Defined as ``(y_pred - target)**2``.
    """

    _tags = {
        "metric_type": "point",
        "info:metric_name": "RMSE",
        "requires:data_type": "point_forecast",
    }  # noqa: E501

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.point import RMSE

        return RMSE
