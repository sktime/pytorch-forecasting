Data v2
=======

.. warning::
    Please note that the v2 modules are currently in active development and are in beta. Please use this API with caution.
    See the complete v2 API documentation :doc:`here <api_v2>` and the stable v1 documentation :doc:`here <api>`.

.. currentmodule:: pytorch_forecasting

API v2 separates the representation of time series data from the preprocessing and data-loading workflow. The data pipeline consists of two main components:

- **TimeSeries:** A data container that holds tabular time series data together with its schema and metadata.
- **DataModule:** Handles preprocessing, batching, dataloader creation, and the collection of model initialization metadata.

TimeSeries: Data Container
--------------------------

The :py:class:`~data.timeseries._timeseries_v2.TimeSeries` class is the standard input and output data structure in API v2. It stores time series data together with information describing its structure, including time, target, group, and variable roles.

It can be initialized from tabular data such as a pandas DataFrame. An optional future-data DataFrame can also be provided for variables known beyond the historical observation period.

Unlike a conventional preprocessing layer, ``TimeSeries`` primarily acts as a data container. It does not perform the full preprocessing and batching workflow handled by the DataModule.

The same data structure is used for forecast output. A prediction result represented as a ``TimeSeries`` can be converted into a pandas DataFrame using ``to_pandas()``:

.. code-block:: python

    predictions_df = predictions.to_pandas()

.. autoclass:: pytorch_forecasting.data.timeseries._timeseries_v2.TimeSeries
   :noindex:
   :members: __init__


DataModule Layer
----------------

The DataModule layer sits on top of ``TimeSeries`` and is implemented using PyTorch Lightning's ``LightningDataModule``. It is responsible for preparing data for model training and prediction.

Its main responsibilities include:

* **Preprocessing:** Applying normalizers and encoders to the data.
* **Batching:** Creating and managing the ``train_dataloader``, ``val_dataloader``, and ``test_dataloader``.
* **Model Initialization Metadata:** Collecting structural information, such as categorical variable counts, embedding sizes, and vocabulary states, required to initialize forecasting models.

**Reusing a DataModule Configuration**

The ``with_data()`` method makes it possible to attach a different ``TimeSeries`` object while retaining the existing DataModule configuration.

It returns a new DataModule instance of the same class with the same configuration. The original DataModule is not modified.

For example:

.. code-block:: python

    new_datamodule = datamodule.with_data(new_time_series)

This is useful when applying an already configured data pipeline to another dataset without rebuilding the configuration.

**Model Compatibility**

Different forecasting architectures require specific input shapes and structures, such as standard sequential batches or encoder-decoder batches. API v2 therefore provides different DataModule implementations.

Each model is designed to work with one or more compatible DataModules. You can check the compatibility overview table in the **:doc:`v2 Models <models_v2>`** documentation to identify the appropriate DataModule for a model.

.. autoclass:: pytorch_forecasting.data.data_module._tslib_data_module.TslibDataModule
   :noindex:
   :members: __init__


API Reference
-------------

See the detailed API documentation for the v2 data classes below:

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
   data.timeseries._timeseries_v2.TimeSeries
   data.data_module._encoder_decoder_data_module.EncoderDecoderTimeSeriesDataModule
   data.data_module._tslib_data_module.TslibDataModule
