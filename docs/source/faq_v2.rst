FAQ v2
======

.. warning::
    Please note that the v2 modules are currently in active-development and are in
    beta right now, so please use this API with caution.
    See complete documentation for v2 API :doc:`here <api_v2>` and stable v1
    documentation :doc:`here <api>`.

.. currentmodule:: pytorch_forecasting

Common questions for users and contributors working with the experimental v2 API.


What is the architecture of v2?
-------------------------------

V2 separates forecasting workflows into four layers:

* **D1 Layer (Dataset):** ``TimeSeries`` ingests raw tabular data and turns it
  into tensors and base metadata. It deliberately avoids the heavier
  preprocessing and batching responsibilities from v1. See :doc:`Data v2
  <data_v2>`.
* **D2 Layer (DataModule):** Lightning data modules such as
  ``EncoderDecoderTimeSeriesDataModule`` and ``TslibDataModule`` sit on top of
  D1. They handle preprocessing, train/validation/test splits, dataloaders, and
  metadata needed to initialize models. See :doc:`Data v2 <data_v2>`.
* **M Layer (Model):** The model layer contains the PyTorch Lightning model and
  the neural network logic. It is designed to be decoupled from the concrete
  data ingestion classes. See :doc:`M Layer v2 <m_layer_v2>`.
* **P Layer (Package):** Package classes wrap the model, compatible data
  module, and Lightning trainer behind a higher-level ``fit`` and ``predict``
  interface. See :doc:`Package Layer v2 <pkg_v2>`.

The full overview is in :doc:`API v2 <api_v2>`.


How do I create a TimeSeries object?
------------------------------------

Create a D1 ``TimeSeries`` object from a pandas ``DataFrame`` and identify the
time, target, group, and feature columns:

.. code-block:: python

    from pytorch_forecasting.data.timeseries import TimeSeries

    dataset = TimeSeries(
        data=data_df,
        time="time_idx",
        target="y",
        group=["series_id"],
        num=["x", "future_known_feature", "static_feature"],
        cat=["category", "static_feature_cat"],
        known=["future_known_feature"],
        unknown=["x", "category"],
        static=["static_feature", "static_feature_cat"],
    )

Use ``data_future`` when future rows are available separately. See
:py:class:`~pytorch_forecasting.data.timeseries._timeseries_v2.TimeSeries` and
:doc:`Data v2 <data_v2>` for constructor details.


What is the difference between TimeSeries, TimeSeriesDataSet, and DataModule?
----------------------------------------------------------------------------

``TimeSeries`` is the v2 D1 dataset. It stores raw time series data, exposes
items as tensors, and records basic metadata. It is intentionally lightweight.

``TimeSeriesDataSet`` is the v1 dataset. It combines data ingestion,
preprocessing, indexing, normalization, and dataloader creation in one class.
Existing stable v1 workflows continue to use it.

V2 data modules are the D2 layer. They consume a D1 ``TimeSeries`` object and
own preprocessing, splits, batching, dataloaders, and model initialization
metadata. This is the main architectural difference for users moving from v1:
responsibilities that used to live together in ``TimeSeriesDataSet`` are split
between D1 and D2 in v2.


How do I create and use data modules?
-------------------------------------

Instantiate the data module that matches the model family and pass the D1
``TimeSeries`` object as ``time_series_dataset``:

.. code-block:: python

    from pytorch_forecasting.data.data_module import (
        EncoderDecoderTimeSeriesDataModule,
    )

    data_module = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=dataset,
        max_encoder_length=30,
        max_prediction_length=1,
        batch_size=32,
    )

The data module can then be passed to a Lightning ``Trainer`` with an M-layer
model, or passed directly to compatible package-layer ``fit`` and ``predict``
methods. Package classes can also build the data module for you when you pass a
D1 ``TimeSeries`` object and provide ``datamodule_cfg``. See :doc:`Data v2
<data_v2>` and :doc:`Models v2 <models_v2>`.


How do I use package classes for fit and predict?
-------------------------------------------------

Package classes provide the highest-level v2 workflow. Configure the data
module, model, and trainer with dictionaries, instantiate a package class, then
call ``fit`` and ``predict``:

.. code-block:: python

    from pytorch_forecasting.metrics import MAE, SMAPE
    from pytorch_forecasting.models.temporal_fusion_transformer._tft_pkg_v2 import (
        TFT_pkg_v2,
    )

    model_pkg = TFT_pkg_v2(
        datamodule_cfg=dict(
            max_encoder_length=30,
            max_prediction_length=1,
            batch_size=32,
        ),
        model_cfg=dict(
            loss=MAE(),
            logging_metrics=[MAE(), SMAPE()],
            optimizer="adam",
            optimizer_params={"lr": 1e-3},
            hidden_size=64,
            num_layers=2,
        ),
        trainer_cfg=dict(max_epochs=5, accelerator="auto", devices=1),
    )

    model_pkg.fit(dataset)
    predictions = model_pkg.predict(dataset, return_info=["index", "x", "y"])

The same package methods can accept a compatible D2 data module where supported.
See :doc:`Package Layer v2 <pkg_v2>` for the full workflow.


How do I add a new model to the package?
----------------------------------------

Start from the v2 extension templates in ``extension_templates/v2``. A new model
usually needs:

* an M-layer model class that inherits from
  :py:class:`~pytorch_forecasting.models.base._base_model_v2.BaseModel`,
  implements ``forward``, and exposes its package class through ``_pkg``;
* a P-layer package class that inherits from
  :py:class:`~pytorch_forecasting.base._base_pkg.Base_pkg`, defines tags,
  returns the model from ``get_cls``, returns a compatible data module from
  ``get_datamodule_cls``, and provides low-cost test parameters;
* a compatible D2 data module, reusing an existing one when possible and adding
  a new one only when the model requires a different batch structure or metadata.

The template README gives the maintainer-facing checklist for models, package
classes, and data modules: `v2 extension templates
<https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v2>`_.
