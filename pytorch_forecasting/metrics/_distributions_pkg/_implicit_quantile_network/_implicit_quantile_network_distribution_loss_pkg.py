"""
Package container for the Implicit Quantile Network distribution loss metric.
"""

from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class ImplicitQuantileNetworkDistributionLoss_pkg(_BasePtMetric):
    """
    Implicit quantile network distribution loss metric for distribution forecasts.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "implicit_quantile_network",
        "info:metric_name": "ImplicitQuantileNetworkDistributionLoss",
        "requires:data_type": "implicit_quantile_network_distribution_forecast",
        "capability:quantile_generation": True,
        "shape:adds_quantile_dimension": True,
        "info:pred_type": ["distr"],
        "info:y_type": ["numeric"],
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.distributions import (
            ImplicitQuantileNetworkDistributionLoss,
        )

        return ImplicitQuantileNetworkDistributionLoss

    @classmethod
    def get_encoder(cls):
        """
        Returns a TorchNormalizer instance for rescaling parameters.
        """
        return TorchNormalizer(transformation="softplus")

    @classmethod
    def get_metric_test_params(cls):
        """
        Returns test parameters for ImplicitQuantileNetworkDistributionLoss.

        This corresponds to the ``output_size`` parameter in the data preparation
        fixture for testing the ImplicitQuantileNetworkDistributionLoss metric.
        """
        return [{"input_size": 5}]

    @classmethod
    def _get_test_dataloaders(cls, params=None):
        """
        Returns test dataloaders configured for ImplicitQuantileNetworkDistributionLoss.
        """
        return super()._get_test_dataloaders_from(params)
