FAQ v2
======

.. warning::
    Please note that the v2 modules are currently in active-development and is in beta right now, so please use this API with caution.
    See complete documentation for v2 API :doc:`here <api_v2>` and stable v1 documentation :doc:`here <api>`.

.. currentmodule:: pytorch_forecasting

Frequently asked questions about API-v2. Other places to seek help from:

* :doc:`v2 API Reference <api_v2>`
* :ref:`v2 Tutorials <tutorials_v2>`
* `API-v2 Development Issue <https://github.com/sktime/pytorch-forecasting/issues/1736>`_
* :doc:`v1 FAQ <faq>` for questions about the stable v1 API


Architecture
------------

* **What is the architecture of v2?**

  API-v2 organizes the entire training and prediction workflow into a strict,
  four-layered structure. Each layer has a single responsibility and can be
  used (or bypassed) independently:

  #. **D1 Layer (Dataset):** the foundational data ingestion layer. It accepts
     raw tabular data (e.g., a pandas DataFrame), converts it into PyTorch
     tensors, and extracts base-level metadata such as static variables. It is
     implemented by :py:class:`~data.timeseries._timeseries_v2.TimeSeries`.
     See :doc:`Data v2 <data_v2>`.
  #. **D2 Layer (DataModule):** a PyTorch Lightning ``LightningDataModule``
     that sits on top of D1. It applies preprocessing (normalizers, encoders),
     creates the train/validation/test dataloaders, and collects the metadata
     (e.g., number of categorical variables, embedding sizes) needed to
     initialize models. See :doc:`Data v2 <data_v2>`.
  #. **M Layer (Model):** the core ``torch`` neural network implementation,
     inheriting from PyTorch Lightning's ``LightningModule``. It is entirely
     agnostic to the data ingestion pipeline. See :doc:`M Layer v2 <m_layer_v2>`.
  #. **P Layer (Package):** a high-level wrapper that orchestrates the other
     layers. It exposes the unified ``fit()`` and ``predict()`` interface and
     houses the fixtures used for testing. See :doc:`Package Layer v2 <pkg_v2>`.

  The main design goal of this layering is to decouple models from data
  structures: in v1, models were tightly coupled to
  :py:class:`~data.timeseries.TimeSeriesDataSet`, whereas v2 models interface
  with standard PyTorch tensors and dataloaders.


Data handling
-------------

* **How do I create a** :py:class:`~data.timeseries._timeseries_v2.TimeSeries` **object?**

  Pass your pandas DataFrame together with the column roles. Only the roles
  you specify explicitly are required - sensible defaults are inferred for
  the rest:

  .. code-block:: python

      from pytorch_forecasting.data.timeseries._timeseries_v2 import TimeSeries

      dataset = TimeSeries(
          data=df,                    # pandas DataFrame with your series
          time="time_idx",            # integer time index column
          target="target",            # column(s) to forecast
          group=["series_id"],        # columns identifying each series
          num=["price", "volume"],    # numerical features
          cat=["category"],           # categorical features
          known=["price"],            # known in the future
          unknown=["volume"],         # not known in the future
          static=["category"],        # constant per series
      )

  Optionally, ``data_future`` can be passed to supply known future covariate
  values. Unlike the v1 ``TimeSeriesDataSet``, the ``TimeSeries`` class does
  not perform scaling, encoding, or windowing - that is the job of the D2
  DataModule.

* **What is the difference between** ``TimeSeries`` **,** ``TimeSeriesDataSet`` **(v1), and a datamodule?**

  This is important to understand when transitioning from v1 to v2:

  * :py:class:`~data.timeseries.TimeSeriesDataSet` (v1) is a single class that
    does everything: tensor conversion, preprocessing, normalization, windowing,
    and dataloader creation. Models in v1 are initialized directly from it
    via ``from_dataset()``.
  * :py:class:`~data.timeseries._timeseries_v2.TimeSeries` (v2, D1 layer) only
    ingests raw data and converts it to tensors plus metadata. It is
    lightweight and does no preprocessing.
  * A **datamodule** (v2, D2 layer), such as
    :py:class:`~data.data_module._encoder_decoder_data_module.EncoderDecoderTimeSeriesDataModule`,
    wraps a ``TimeSeries`` object and performs the preprocessing, batching,
    and metadata collection that ``TimeSeriesDataSet`` used to do in v1.

  In short: v1's ``TimeSeriesDataSet`` roughly equals v2's
  ``TimeSeries`` + a datamodule.

* **How do I create and use a datamodule?**

  Wrap your D1 ``TimeSeries`` object in the datamodule that matches your
  model family, and pass windowing/batching parameters to it:

  .. code-block:: python

      from pytorch_forecasting.data.data_module._encoder_decoder_data_module import (
          EncoderDecoderTimeSeriesDataModule,
      )

      datamodule = EncoderDecoderTimeSeriesDataModule(
          time_series_dataset=dataset,   # a D1 TimeSeries object
          max_encoder_length=30,
          max_prediction_length=7,
          batch_size=32,
      )

  Different model architectures require different input structures, so v2
  provides several datamodules (e.g.,
  :py:class:`~data.data_module._encoder_decoder_data_module.EncoderDecoderTimeSeriesDataModule`
  and :py:class:`~data.data_module._tslib_data_module.TslibDataModule`).
  Check the compatibility overview in :doc:`Models v2 <models_v2>` to see
  which datamodule pairs with your chosen model.

  When using the Package layer, you usually do not instantiate a datamodule
  yourself - you pass a ``datamodule_cfg`` dictionary and a ``TimeSeries``
  object instead, and the package constructs the datamodule internally.


Training and prediction
-----------------------

* **How do I use** ``pkg`` **classes to fit and predict?**

  Package classes provide a ``scikit-learn``-like workflow driven by three
  configuration dictionaries - no Lightning boilerplate required:

  .. code-block:: python

      from pytorch_forecasting.models.temporal_fusion_transformer._tft_pkg_v2 import TFT_pkg_v2
      from pytorch_forecasting.metrics import MAE, SMAPE

      datamodule_cfg = dict(
          max_encoder_length=30,
          max_prediction_length=1,
          batch_size=32,
      )
      model_cfg = dict(
          loss=MAE(),
          logging_metrics=[MAE(), SMAPE()],
          optimizer="adam",
          optimizer_params={"lr": 1e-3},
          hidden_size=64,
      )
      trainer_cfg = dict(
          max_epochs=5,
          accelerator="auto",
      )

      model_pkg = TFT_pkg_v2(
          model_cfg=model_cfg,
          trainer_cfg=trainer_cfg,
          datamodule_cfg=datamodule_cfg,
      )

      model_pkg.fit(dataset)   # dataset is a D1 TimeSeries object
      preds = model_pkg.predict(dataset, return_info=["index", "x", "y"])

  ``fit()`` and ``predict()`` accept a D1 ``TimeSeries`` (wrapped in a D2
  datamodule internally), a pre-configured D2 datamodule, or - for
  ``predict()`` - a plain PyTorch ``DataLoader``. See
  :doc:`Package Layer v2 <pkg_v2>` for details, and the
  :doc:`v2 walkthrough </tutorials/ptf_V2_example>` for a full example.

* **Do I have to use the Package layer?**

  No. The Package layer is a convenience wrapper. Advanced users can work
  with the M layer directly as a standard PyTorch Lightning ``LightningModule``
  and build fully custom training, validation, and prediction pipelines.
  See :doc:`M Layer v2 <m_layer_v2>` and the :ref:`v2 tutorials <tutorials_v2>`.


Contributing
------------

* **How do I add a new model to the package?**

  #. Start from the `v2 extension templates
     <https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v2>`_,
     which walk through the required structure for the model and its datamodule.
  #. Implement the network as an M-layer class inheriting from the v2 base model
     in :py:mod:`~models.base._base_model_v2`.
  #. Add a corresponding package (P-layer) class inheriting from
     :py:class:`~base._base_pkg.Base_pkg`, which registers the model's tags
     and test fixtures.
  #. Check the existing v2 models (e.g., TFT, DLinear, TimeXer) for reference
     implementations, and open a pull request - contributions are welcome!
