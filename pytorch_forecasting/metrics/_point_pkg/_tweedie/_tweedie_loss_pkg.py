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
        "info:pred_type": ["point"],
        "info:y_types": ["numeric"],
        "loss_ndim": 1,
    }  # noqa: E501

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.point import TweedieLoss

        return TweedieLoss

    @classmethod
    def _get_test_dataloaders(cls, params=None):
        """
        Returns test dataloaders configured for TweedieLoss.
        """
        return super()._get_test_dataloaders_from(params, target="agency")
