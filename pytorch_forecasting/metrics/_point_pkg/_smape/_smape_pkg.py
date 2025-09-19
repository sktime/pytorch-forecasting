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

    _tags = {
        "metric_type": "point",
        "info:metric_name": "SMAPE",
        "requires:data_type": "point_forecast",
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
        "loss_ndim": 1,
    }  # noqa: E501

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.point import SMAPE

        return SMAPE

    @classmethod
    def _get_test_dataloaders(cls, params=None):
        """
        Returns test dataloaders configured for SMAPE.
        """
        return super()._get_test_dataloaders_from(params=params, target="agency")
