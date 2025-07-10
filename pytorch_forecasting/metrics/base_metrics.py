"""
Re-export of base metrics functionality from `base/_base_metrics.py`.
Maintains compatibility with the previous structure while allowing for
future extensions or modifications in the base metrics implementation.
"""

from pytorch_forecasting.metrics.base._base_metrics import (
    AggregationMetric,
    CompositeMetric,
    DistributionLoss,
    Metric,
    MultiHorizonMetric,
    MultiLoss,
    MultivariateDistributionLoss,
    TorchMetricWrapper,
    convert_torchmetric_to_pytorch_forecasting_metric,
)

# Re-export all symbols
__all__ = [
    "Metric",
    "MultiHorizonMetric",
    "DistributionLoss",
    "MultivariateDistributionLoss",
    "MultiLoss",
    "convert_torchmetric_to_pytorch_forecasting_metric",
    "AggregationMetric",
    "TorchMetricWrapper",
    "CompositeMetric",
]
