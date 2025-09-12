"""
Package container for the Mean Absolute Scaled Error (MASE) metric for point forecasts.
"""

from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class MASE_pkg(_BasePtMetric):
    """
    Mean Average scaled Error (MASE) metric for point forecasts.
    """

    _tags = {
        "metric_type": "point",
        "info:metric_name": "MASE",
        "requires:data_type": "point_forecast",
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics import MASE

        return MASE

    @classmethod
    def _get_test_dataloaders_from(cls, params=None):
        """
        Returns test dataloaders configured for MASE.
        """
        return super()._get_test_dataloaders_from(params=params, target="agency")
