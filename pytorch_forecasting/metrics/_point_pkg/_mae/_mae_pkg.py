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
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
        "loss_ndim": 1,
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics import MAE

        return MAE

    @classmethod
    def _get_test_dataloaders(cls, params=None):
        """
        Returns test dataloaders configured for MAE.
        """
        return super()._get_test_dataloaders_from(params=params, target="agency")
