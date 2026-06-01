Data
====

.. important::
   **Try the New API v2!**

   We are currently building the next generation of PyTorch Forecasting. You are reading the legacy documentation for the v1 data pipelines.
   If you would like to test the new architecture, check out the documentation of Data v2 :doc:`here <data_v2>` and complete API v2 documentation :doc:`here <api_v2>`.
   **Please note that API v2 is currently a Work in Progress and is considered unstable, so use it with caution in production environments.**

.. currentmodule:: pytorch_forecasting.data

Loading data for timeseries forecasting is not trivial - in particular if covariates are included and values are missing.
PyTorch Forecasting provides the :py:class:`~timeseries.TimeSeriesDataSet` which comes with a :py:meth:`~timeseries.TimeSeriesDataSet.to_dataloader`
method to convert it to a dataloader and a :py:meth:`~timeseries.TimeSeriesDataSet.from_dataset` method to create, e.g. a validation
or test dataset from a training dataset using the same label encoders and data normalization.

Further, timeseries have to be (almost always) normalized for a neural network to learn efficiently. PyTorch Forecasting
provides multiple such target normalizers (some of which can also be used for normalizing covariates).


Time series data set
---------------------

The time series dataset is the central data-holding object in PyTorch Forecasting. It primarily takes
a pandas DataFrame along with some metadata. See the :ref:`tutorial on passing data to models <passing-data>` to learn more it is coupled to models.

.. autoclass:: pytorch_forecasting.data.timeseries.TimeSeriesDataSet
   :noindex:
   :members: __init__

Details
--------

See the API documentation for further details on available data encoders and the :py:class:`~timeseries.TimeSeriesDataSet`:

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api

    data.encoders.EncoderNormalizer
    data.encoders.GroupNormalizer
    data.encoders.MultiNormalizer
    data.encoders.NaNLabelEncoder
    data.encoders.TorchNormalizer
    data.samplers.TimeSynchronizedBatchSampler
    data.samplers.GroupedSampler
    data.timeseries.TimeSeriesDataSet
