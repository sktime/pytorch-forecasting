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
        "compatible_pred_types": ["point"],
        "compatible_y_types": ["category"],
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics import CrossEntropy

        return CrossEntropy

    @classmethod
    def _get_test_dataloaders_from(cls, params=None):
        """
        Returns test dataloaders configured for CrossEntropy.
        """
        from pytorch_forecasting.tests._data_scenarios import data_with_covariates, make_dataloaders

        if params is None:
            params = {}
        data_loader_kwargs = params.get("data_loader_kwargs", {})
        # For classification, set target to a categorical column, e.g., "agency"
        data_loader_kwargs.setdefault("target", "agency")

        data = data_with_covariates()
        dataloaders = make_dataloaders(data, **data_loader_kwargs)
        return dataloaders
