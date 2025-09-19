"""
Package container for the Mean Absolute Percentage Error (MAPE) metric for point
forecasts.
"""

from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class MAPE_pkg(_BasePtMetric):
    """
    Mean absolute percentage error metric for point forecasts.

    Defined as ``(y - y_pred).abs() / y.abs()``.
    Assumes ``y >= 0``.
    """

    _tags = {
        "metric_type": "point",
        "info:metric_name": "MAPE",
        "requires:data_type": "point_forecast",
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
        "loss_ndim": 1,
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.point import MAPE

        return MAPE

    @classmethod
    def _get_test_dataloaders(cls, params=None):
        """
        Returns test dataloaders configured for MAPE.
        """
        return super()._get_test_dataloaders_from(params=params, target="agency")
