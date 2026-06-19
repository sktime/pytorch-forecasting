Data v2
=======

.. warning::
    Please note that the v2 modules are currently in active-development and is in beta right now, so please use this API with caution.
    See complete documentation for v2 API :doc:`here <api_v2>` and stable v1 documentation :doc:`here <api>`.

.. currentmodule:: pytorch_forecasting

Loading and managing time series data for deep learning can be complex, especially when handling varying sequence lengths, multiple covariates, and categorical encodings.

In API-v2, the data pipeline follows a strict two-layer architecture: the **D1 Layer (Dataset)** and the **D2 Layer (DataModule)** to maintain "separation of responsibilities".

- **D1 Layer (Dataset)** ingests the raw data and turn it into ``torch`` tensors
- **D2 Layer (DataModule)** performs the pre-processing and the data loading


D1 Layer: Dataset
-----------------

The D1 Layer is the foundational data ingestion layer. Its primary responsibilities are to accept raw tabular data (e.g., pandas DataFrames), convert the raw data into PyTorch tensors, and extract base-level metadata such as static variables and basic time series properties.

Unlike the v1 dataset, the D1 layer does not handle complex preprocessing or batching logic, keeping it lightweight and highly modular.

.. autoclass:: pytorch_forecasting.data.timeseries._timeseries_v2.TimeSeries
   :noindex:
   :members: __init__


D2 Layer: DataModule
--------------------

The D2 Layer sits on top of D1 and is implemented as a PyTorch Lightning ``LightningDataModule``. This layer is responsible for the heavier lifting:

* **Preprocessing:** Applying normalizers and encoders to the data.
* **Batching:** Creating and managing the ``train_dataloader``, ``val_dataloader``, and ``test_dataloader``.
* **Model Initialization Metadata:** Dynamically collecting necessary architectural information (such as the number of categorical variables, embedding sizes, and vocabulary states) required to properly instantiate the Forecasting models in the Model Layer.


**Model Compatibility**

Because different forecasting architectures require specific input shapes and structures (e.g., standard sequential batches vs. complex encoder-decoder structures), there are several different types of DataModules available in API-v2.

Each model is optimally designed to be compatible with one or more specific DataModules. You can easily verify which DataModule pairs correctly with your chosen model by checking the compatibility overview table in the **:doc:`v2 Models <models_v2>`** documentation.

.. autoclass:: pytorch_forecasting.data.data_module._tslib_data_module.TslibDataModule
   :noindex:
   :members: __init__


API Reference
-------------

See the detailed API documentation for the V2 data classes below:

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
