"""
Package container for Poisson distribution loss metric for point forecasts.
"""

from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class PoissonLoss_pkg(_BasePtMetric):
    """
    Poisson loss for count data.

    The loss will take the exponential of the network output before it is returned as prediction.
    """  # noqa: E501

    _tags = {
        "metric_type": "point",
        "info:metric_name": "PoissonLoss",
        "requires:data_type": "point_forecast",
        "capability:quantile_generation": True,
        "shape:adds_quantile_dimension": True,
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.point import PoissonLoss

        return PoissonLoss

    @classmethod
    def _get_test_dataloaders_from(cls, params=None):
        """
        Returns test dataloaders configured for PoissonLoss.
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
