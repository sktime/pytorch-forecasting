Data
====

.. currentmodule:: pytorch_forecasting.data

This module provides data loading, preprocessing, and dataset utilities for time series pytorch_forecasting
Objects are organized by their primary use case.

Neural Networks / Dataset Classes
---------------------------------

.. autosummary::
    :toctree: ../api

    TimeSeriesDataSet
    NumpyDataset

Package Containers
------------------

.. autosummary::
    :toctree: ../api

    DataLoader

utilities
---------

.. autosummary::
    :toctree: ../api

    create_lagged_data
    create_rolling_window
