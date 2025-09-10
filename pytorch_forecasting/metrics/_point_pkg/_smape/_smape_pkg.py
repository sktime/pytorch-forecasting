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
    }  # noqa: E501

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.point import SMAPE

        return SMAPE

    @classmethod
    def _get_test_dataloaders_from(cls, params=None):
        """
        Returns test dataloaders configured for SMAPE.
        """
        from pytorch_forecasting.tests._data_scenarios import (
            data_with_covariates,
            make_dataloaders,
        )

        if params is None:
            params = {}
        data_loader_kwargs = params.get("data_loader_kwargs", {})
        # For point metrics, default target is "target"
        data_loader_kwargs.setdefault("target", "target")

        data = data_with_covariates()
        dataloaders = make_dataloaders(data, **data_loader_kwargs)
        return dataloaders
