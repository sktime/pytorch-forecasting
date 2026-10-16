"""
Base classes for pytorch-forecasting metrics.
"""

from pytorch_forecasting.metrics.base_metrics._base_metrics import (
    AggregationMetric,
    CompositeMetric,
    DistributionLoss,
    Metric,
    MultiHorizonMetric,
    MultiLoss,
    MultivariateDistributionLoss,
    coerce_to_pytorch_forecasting_metric,
)

__all__ = [
    "Metric",
    "MultiHorizonMetric",
    "DistributionLoss",
    "MultivariateDistributionLoss",
    "MultiLoss",
    "coerce_to_pytorch_forecasting_metric",
    "AggregationMetric",
    "CompositeMetric",
]
