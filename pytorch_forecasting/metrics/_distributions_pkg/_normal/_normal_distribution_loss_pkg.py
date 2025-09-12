"""
Package container for the Normal distribution loss metric.
"""

from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class NormalDistributionLoss_pkg(_BasePtMetric):
    """
    Normal distribution loss metric for distribution forecasts.

    Defined as ``(y_pred - target)**2``.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "normal",
        "info:metric_name": "NormalDistributionLoss",
        "requires:data_type": "normal_distribution_forecast",
        "info:pred_type": ["distr"],
        "info:y_type": ["numeric"],
        "expected_loss_ndim": 2,
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.distributions import NormalDistributionLoss

        return NormalDistributionLoss

    @classmethod
    def _get_test_dataloaders_from(cls, params=None):
        """
        Returns test dataloaders configured for NormalDistributionLoss.
        """
        super()._get_test_dataloaders_from(params=params, target="agency")
