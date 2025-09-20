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
        "loss_ndim": 1,
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.point import PoissonLoss

        return PoissonLoss

    @classmethod
    def _get_test_dataloaders(cls, params=None):
        """
        Returns test dataloaders configured for PoissonLoss.
        """
        return super()._get_test_dataloaders_from(params=params, target="agency")
