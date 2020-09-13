Models
======

.. _models:

.. currentmodule:: pytorch_forecasting

Model parameters very much depend on the dataset for which they are destined.

Pytorch Forecasting provides a ``.from_dataset()`` method for each model that
takes a :py:class:`~data.timeseries.TimeSeriesDataSet` and additional parameters
that cannot directy derived from the dataset such as, e.g. ``learning_rate`` or ``hidden_size``.

To tune models, `optuna <https://optuna.readthedocs.io/>`_ can be used. For example, tuning of the :py:class:`~models.temporal_fusion_transformer.TemporalFusionTransformer`
is implemented by :py:func:`~models.temporal_fusion_transformer.tuning.optimize_hyperparameters`

Details
--------

See the API documentation for further details on available models:

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api/
   :template: custom-module-template.rst
   :recursive:

   models
