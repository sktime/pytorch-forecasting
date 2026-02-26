Metrics
=======

.. currentmodule:: pytorch_forecasting.metrics

This module provides loss functions and evaluation metrics for time series forecasting.
Metrics are organized by their application context.

Point / Loss Functions
-----------------------
.. note::

   Some internal ``*_pkg`` metric classes (e.g. ``MAE_pkg``, ``RMSE_pkg``) are used
   for packaging and internal integration. These are not part of the public API
   and are therefore not documented here.

.. autosummary::
   :toctree: ../api

   QuantileLoss
   MAE
   MAPE
   MASE
   RMSE
   SMAPE
   CrossEntropy
   PoissonLoss
   TweedieLoss

Distribution Losses
------------------

.. note::

   Internal ``*_pkg`` variants (e.g. ``NormalDistributionLoss_pkg``) are used
   for packaging and internal integration and are not part of the public API.

.. autosummary::
   :toctree: ../api

   DistributionLoss
   MultivariateDistributionLoss
   BetaDistributionLoss
   NegativeBinomialDistributionLoss
   NormalDistributionLoss
   LogNormalDistributionLoss
   MultivariateNormalDistributionLoss
   ImplicitQuantileNetworkDistributionLoss
   MQF2DistributionLoss

Base Classes
------------
.. note::

   ``DistributionLoss`` and ``MultivariateDistributionLoss`` are documented
   under *Distribution Losses* for clarity, although they are base classes.

.. autosummary::
   :toctree: ../api

   Metric
   MultiHorizonMetric
   MultiLoss
   convert_torchmetric_to_pytorch_forecasting_metric

Composite and Aggregation (base_metrics)
----------------------------------------

.. currentmodule:: pytorch_forecasting.metrics.base_metrics

.. autosummary::
   :toctree: ../api

   CompositeMetric
   AggregationMetric
