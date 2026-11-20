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


Comparison with pytorch-timeseries
----------------------------------

The `pytorch-timeseries dataset API`_ exposes a numeric matrix and timestamps
for benchmark datasets such as ETTh1. PyTorch Forecasting's ``TimeSeries``
accepts a DataFrame with an explicit schema for series groups, targets, and
known, unknown, and static covariates. Both projects separate the source
data from task-specific windowing and batching.

.. list-table::
   :header-rows: 1
   :widths: 21 39 40

   * - Design choice
     - pytorch-timeseries
     - PyTorch Forecasting v2
   * - Train and evaluation splits
     - `ForecastDataModule`_ splits observations chronologically using
       explicit borders, dataset defaults, or ratios. Evaluation windows
       can use history from before a split boundary. The scaler is fitted
       on the training subset.
     - `BaseTimeSeriesDataModule`_ assigns whole series to splits when
       there are enough groups. A single series is used in all three
       splits. Group-based splits measure generalization to unseen
       series; they do not measure forecasting on held-out future
       observations of the same series.
   * - Windows and batches
     - ``WindowConfig`` sets the lookback, gap before the forecast, forecast
       length, stride, and input and target columns. The loader yields a
       named batch with fields such as ``x`` and ``y``.
     - ``TslibDataModule`` uses context length, prediction length, and stride
       to create windows within each series. The datamodule provides
       model-specific metadata and batches for Lightning.
   * - Model contract
     - For example, `pytorch-timeseries DLinear`_ takes a batched input
       tensor and returns a forecast tensor in regression mode.
     - `DLinearForecaster`_ pairs DLinear with ``TslibDataModule`` and uses
       datamodule metadata to initialize the model. A shared architecture
       name does not imply compatible model inputs or batch formats.

The two data pipelines have different schema, split, and model contracts.
A direct dependency would still require adapting those contracts. The most
relevant design to carry forward is an opt-in chronological split for
evaluating future observations of the same series. The existing
`split-strategy proposal`_ tracks that work. Its evaluation windows should
allow past observations as context while keeping targets within the
evaluation partition and fitting transforms only on training data.

.. _pytorch-timeseries dataset API:
   https://github.com/wayne155/pytorch_timeseries/blob/28f0424b60cb0a69140356c63b6862f417aa315a/docs/modules/dataset.rst
.. _ForecastDataModule:
   https://github.com/wayne155/pytorch_timeseries/blob/28f0424b60cb0a69140356c63b6862f417aa315a/torch_timeseries/dataloader/v2/forecast.py
.. _BaseTimeSeriesDataModule:
   https://github.com/sktime/pytorch-forecasting/blob/570ab4047fd6537dd05cb05cf74bb973c322a4e9/pytorch_forecasting/data/data_module/base_data_module.py#L296-L317
.. _pytorch-timeseries DLinear:
   https://github.com/wayne155/pytorch_timeseries/blob/28f0424b60cb0a69140356c63b6862f417aa315a/torch_timeseries/model/DLinear.py
.. _DLinearForecaster:
   https://github.com/sktime/pytorch-forecasting/blob/570ab4047fd6537dd05cb05cf74bb973c322a4e9/pytorch_forecasting/models/dlinear/_dlinear_forecaster_v2.py
.. _split-strategy proposal:
   https://github.com/sktime/pytorch-forecasting/pull/2087


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
