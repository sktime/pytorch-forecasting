Metrics
==========

Multiple metrics have been implemented to ease adaptation.

In particular, these metrics can be applied to the multi-horizon forecasting problem, i.e.
can take tensors that are not only of shape ``n_samples`` but also ``n_samples x prediction_horizon``
or even ``n_samples x prediction_horizon x n_outputs``, where ``n_outputs`` could be the number
of forecasted quantiles.

Metrics can be easily combined by addition, e.g.

.. code-block:: python

   from pytorch_forecasting.metrics import SMAPE, MAE

   composite_metric = SMAPE() + 1e-4 * MAE()

Such composite metrics are useful when training because they can reduce outliers in other metrics.
In the example, SMAPE is mostly optimized, while large outliers in MAE are avoided.

Further, one can modify a loss metric to reduce a mean prediction bias, i.e. ensure that
predictions add up. For example:

.. code-block:: python

   from pytorch_forecasting.metrics import MAE, AggregationMetric

   composite_metric = MAE() + AggregationMetric(metric=MAE())

Here we add to MAE an additional loss. This additional loss is the MAE calculated on the mean predictions
and actuals. We can also use other metrics such as SMAPE to ensure aggregated results are unbiased in that metric.
One important point to keep in mind is that this metric is calculated accross samples, i.e. it will vary depending
on the batch size. In particular, errors tend to average out with increased batch sizes.


Details
--------

See the API documentation for further details on available metrics:

.. currentmodule:: pytorch_forecasting

.. moduleautosummary::
   :toctree: api
   :template: custom-module-template.rst
   :recursive:

   pytorch_forecasting.metrics
