Model overview
==============

.. warning::
    The v2 API is in beta and unstable. It may change without warning and is not
    ready for production use.

The table lists every model in ``pytorch-forecasting``. Each name links to the
forecaster class you train with. "Compatible Data Modules" is the datamodule the
forecaster builds when you do not pass one. To use another configuration of it,
pass your own instance as ``datamodule``, see :doc:`data_v2`.

.. model-overview-v2::

The columns come from the tags each forecaster declares.

* Covariates: the model uses features besides the target.
* Multiple targets: the model can forecast several targets at once.
* Regression: the model supports numerical targets.
* Classification: the model supports categorical targets.
* Probabilistic: the model predicts a distribution.
* Prediction intervals: the model can produce prediction intervals.
* Flexible History Length: the model works with a history of varying length.
* Cold Start: the model can predict series with little or no history.
* Compute (1-5): rough compute cost, where 5 is the most expensive.
