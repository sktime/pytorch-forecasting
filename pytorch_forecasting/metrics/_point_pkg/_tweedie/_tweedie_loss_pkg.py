"""
Package container for the Tweedie loss metric for point forecasts.
"""

from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class TweedieLoss_pkg(_BasePtMetric):
    """
    Tweedie loss for regression with exponential dispersion models.

    Tweedie regression with log-link. Useful for modeling targets that might be tweedie-distributed.
    """  # noqa: E501

    _tags = {
        "metric_type": "point",
        "info:metric_name": "TweedieLoss",
        "requires:data_type": "point_forecast",
        "compatible_pred_types": ["point"],
        "compatible_y_types": ["numeric"],
    }  # noqa: E501

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.point import TweedieLoss

        return TweedieLoss

    @classmethod
    def _get_test_dataloaders_from(cls, params=None):
        """
        Returns test dataloaders configured for TweedieLoss.
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
