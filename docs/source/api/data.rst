Data
====

.. currentmodule:: pytorch_forecasting.data

This module provides data loading, preprocessing, and dataset utilities for time series forecasting.
Objects are organized by their primary use case.

Datasets
--------
.. note::

   These classes form the core data handling API for PyTorch Forecasting.

.. autosummary::
    :toctree: ../api

    TimeSeriesDataSet

Encoders / Normalizers
----------------------

.. autosummary::
    :toctree: ../api

    NaNLabelEncoder
    TorchNormalizer
    EncoderNormalizer
    GroupNormalizer
    MultiNormalizer

Samplers
--------

.. autosummary::
    :toctree: ../api

    TimeSynchronizedBatchSampler
