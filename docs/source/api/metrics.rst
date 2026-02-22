Metrics
=======

.. currentmodule:: pytorch_forecasting.metrics

This module provides loss functions and evaluation metrics for time series forecasting.
Metrics are organized by their application context.

Neural Networks / Loss Functions
---------------------------------

.. autosummary::
   :toctree: ../api

   QuantileLoss
   MAE
   MASE
   RMSE
   SMAPE

Package Containers
-------------------

.. autosummary::
   :toctree: ../api

   Metric
   CompositeMetric

Utilities
---------

.. autosummary::
   :toctree: ../api

   multivariate_normal_distribution
   aggregate_metrics
