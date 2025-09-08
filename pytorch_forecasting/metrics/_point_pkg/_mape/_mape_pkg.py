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
        "compatible_pred_types": ["point"],
        "compatible_y_types": ["numeric"],
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.point import MAPE

        return MAPE

    @classmethod
    def _get_test_dataloaders_from(cls, params=None):
        """
        Returns test dataloaders configured for MAPE.
        """
        from pytorch_forecasting.tests._data_scenarios import data_with_covariates, make_dataloaders

        if params is None:
            params = {}
        data_loader_kwargs = params.get("data_loader_kwargs", {})
        # For point metrics, default target is "target"
        data_loader_kwargs.setdefault("target", "target")

        data = data_with_covariates()
        dataloaders = make_dataloaders(data, **data_loader_kwargs)
        return dataloaders
