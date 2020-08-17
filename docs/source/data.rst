Data
=====

.. currentmodule:: pytorch_forecasting.data

Loading data for timeseries forecasting is not trivial - in particular if covariates are included and values are missing.
PyTorch Forecasting provides the :py:class:`~TimeSeriesDataSet` which comes with a :py:meth`~TimeSeriesDataSet.to_dataloader`
method to convert it to a dataloader and a :py:meth`~TimeSeriesDataSet.from_dataset` method to create, e.g. a validation
or test dataset from a training dataset using the same label encoders and data normalization.

Further, timeseries have to be (almost always) normalized for a neural network to learn efficiently. PyTorch Forecasting
provides multiple such target normalizers (which can largely also be used for normalizing covariates).

.. include:: _autosummary/pytorch_forecasting.data.rst
    :start-line: 3