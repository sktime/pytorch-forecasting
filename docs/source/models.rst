Models
========

Model parameters very much depend on the dataset for which they are destined.

Pytorch Forecasting provides a ``.from_dataset()`` method for each model that
takes a :py:class:`~pytorch_forecasting.TimeSeriesDataSet` and additional parameters
that cannot directy derived from the dataset such as, e.g. ``learning_rate`` or ``hidden_size``.

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api/
   :template: custom-module-template.rst
   :recursive:

   pytorch_forecasting.models
