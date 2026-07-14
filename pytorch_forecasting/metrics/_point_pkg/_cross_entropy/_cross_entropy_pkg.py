"""
Package container for the Cross Entropy Loss metric for point forecasts.
"""

from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class CrossEntropy_pkg(_BasePtMetric):
    """
    Package container for CrossEntropyLoss - a loss function for categorical targets.

    CrossEntropyLoss is used for multi-class classification tasks where the target is
    categorical.
    """

    _tags = {
        "metric_type": "point_classification",
        "requires:data_type": "classification_forecast",
        "info:metric_name": "CrossEntropy",
        "no_rescaling": True,
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics import CrossEntropy

        return CrossEntropy
