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
        "capability:quantile_generation": True,
        "shape:adds_quantile_dimension": True,
    }

    @classmethod
    def get_model_cls(cls):
        from pytorch_forecasting.metrics.point import PoissonLoss

        return PoissonLoss
