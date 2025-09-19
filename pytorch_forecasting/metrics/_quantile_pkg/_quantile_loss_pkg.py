"""
Package containers for Quantile Loss metric for quantile forecasts.
"""

from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class QuantileLoss_pkg(_BasePtMetric):
    """
    Quantile loss metric for quantile forecasts.

    Defined as ``(y_pred - target).abs()``.
    """

    _tags = {
        "metric_type": "quantile",
        "info:metric_name": "QuantileLoss",
        "requires:data_type": "quantile_forecast",
        "info:pred_type": ["quantile"],
        "info:y_type": ["numeric"],
        "num_quantiles": 2,
    }  # noqa: E501

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics import QuantileLoss

        return QuantileLoss

    @classmethod
    def get_metric_test_params(cls):
        return [
            {
                "quantiles": [0.1, 0.5, 0.9],
            },
            {
                "quantiles": [0.2, 0.5],
            },
        ]

    @classmethod
    def _get_test_dataloaders(cls, params=None):
        """
        Returns test dataloaders configured for QuantileLoss.
        """
        return super()._get_test_dataloaders_from(params, target="agency")
