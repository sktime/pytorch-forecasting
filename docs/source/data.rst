Data
=====

.. currentmodule:: pytorch_forecasting.data

Loading data for timeseries forecasting is not trivial - in particular if covariates are included and values are missing.
PyTorch Forecasting provides the :py:class:`~timeseries.TimeSeriesDataSet` which comes with a :py:meth:`~timeseries.TimeSeriesDataSet.to_dataloader`
method to convert it to a dataloader and a :py:meth:`~timeseries.TimeSeriesDataSet.from_dataset` method to create, e.g. a validation
or test dataset from a training dataset using the same label encoders and data normalization.

Further, timeseries have to be (almost always) normalized for a neural network to learn efficiently. PyTorch Forecasting
provides multiple such target normalizers (some of which can also be used for normalizing covariates).

Details
--------

See the API documentation for further details on available data encoders and the :py:class:`~timeseries.TimeSeriesDataSet`:

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api/
   :template: custom-module-template.rst
   :recursive:
   
   pytorch_forecasting.data


