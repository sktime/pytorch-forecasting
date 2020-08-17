Metrics
==========

Multiple metrics have been implemented to ease adaptation.

In particular, these metrics can be applied to the multi-horizon forecasting problem, i.e.
can take tensors that are not only of shape ``n_samples`` but also ``n_samples x prediction_horizon`` 
or even ``n_samples x prediction_horizon x n_outputs``, where ``n_outputs`` could be the number
of forecasted quantiles.

The following metrics are available

.. include:: _autosummary/pytorch_forecasting.metrics.rst
    :start-line: 3