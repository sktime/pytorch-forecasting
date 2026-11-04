Migration guide: v1 to v2
==========================

.. warning::
   The v2 API is currently a beta, experimental API. It is not yet production
   ready and may change without prior notice. Keep existing v1 pipelines until
   you have validated a v2 migration for your workload.

This guide explains the main differences between the stable v1 API and the
experimental v2 API. It is intended for users who already have a v1 forecasting
pipeline and want to evaluate v2 without losing their existing workflow.

For the complete API references, see the :doc:`v1 API <api>` and
:doc:`v2 API <api_v2>` documentation.


At a glance
-----------

The biggest change in v2 is the separation of data preparation, model logic,
and workflow orchestration:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Concern
     - v1
     - v2
   * - Raw data
     - :class:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`
     - :class:`~pytorch_forecasting.data.timeseries._timeseries_v2.TimeSeries`
       (D1 layer)
   * - Preprocessing and batching
     - Handled by ``TimeSeriesDataSet``
     - A Lightning ``DataModule`` (D2 layer)
   * - Model
     - A Lightning model usually created with ``.from_dataset()``
     - A data-agnostic Lightning model created from D2 metadata (M layer)
   * - High-level workflow
     - ``Trainer.fit(model, train_dataloader, ...)``
     - A v2 package's ``fit()`` and ``predict()`` methods (P layer), or the
       Lightning workflow directly
   * - Configuration
     - Dataset and model arguments are commonly coupled
     - Separate ``datamodule_cfg``, ``model_cfg``, and ``trainer_cfg``


What stays the same
-------------------

* Both APIs use PyTorch and Lightning and can be used with custom Lightning
  trainers and dataloaders.
* The existing metrics and normalizers remain available. v2 models use the
  same metrics package while the v2 interface is being developed.
* The forecasting concepts are unchanged: targets, groups, known and unknown
  variables, encoder windows, prediction windows, and covariates still need to
  be specified.
* v1 and v2 can be installed from the same ``pytorch-forecasting`` release.
  They use different classes and should be migrated incrementally rather than
  mixed in one model pipeline.


What changes
------------

* ``TimeSeriesDataSet`` is replaced by a lightweight v2 ``TimeSeries`` D1
  dataset plus a D2 ``DataModule``. The D1 dataset ingests the tabular data and
  records metadata; the D2 module handles encoders, scalers, splitting, and
  dataloaders.
* v2 models do not receive a v1 dataset. They receive metadata produced by the
  D2 module and operate on dictionaries of tensors.
* The v2 package layer can construct the D2 module, model, and Lightning
  trainer from separate configuration dictionaries.
* v2 model and package class names are not always the same as v1 names. Check
  the :doc:`v2 model overview <models_v2>` before choosing a replacement.
* When a package fit uses its default checkpointing, it saves the model
  checkpoint alongside the model configuration, DataModule configuration, and
  metadata artifacts. Keep these files together when moving a trained v2
  package to another environment. Do not assume that a v1 checkpoint can be
  loaded by a v2 class (or vice versa).


Migration steps
---------------

1. Keep the v1 pipeline working
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do not replace a production v1 pipeline before testing the new pipeline on the
same data, split, metrics, and forecast horizon. v1 remains the stable API;
the :doc:`v1 tutorials <tutorials>` and :doc:`v1 API <api>` continue to apply.

2. Describe the data with the v2 D1 layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The v1 dataset combines data description, preprocessing, windowing, and
dataloaders. In v2, start with ``TimeSeries`` and explicitly describe the
roles of the columns:

.. code-block:: python

   from pytorch_forecasting.data.timeseries import TimeSeries

   dataset = TimeSeries(
       data=data,
       time="time_idx",
       target="y",
       group=["series_id"],
       num=["known_value", "unknown_value"],
       cat=["series_type"],
       known=["known_value"],
       unknown=["unknown_value"],
       static=["series_type"],
   )

The equivalent v1 declaration would normally put these roles and the sequence
configuration in one ``TimeSeriesDataSet`` call:

.. code-block:: python

   from pytorch_forecasting import TimeSeriesDataSet

   dataset = TimeSeriesDataSet(
       data,
       time_idx="time_idx",
       target="y",
       group_ids=["series_id"],
       time_varying_known_reals=["known_value"],
       time_varying_unknown_reals=["unknown_value"],
       static_categoricals=["series_type"],
       max_encoder_length=30,
       max_prediction_length=6,
   )

The v2 names are intentionally shorter, but they are not a mechanical rename
for every v1 dataset option. Review the :doc:`v2 data documentation <data_v2>`
for the supported D1 and D2 parameters.

Column roles are split across two kinds of v2 arguments: ``num`` and ``cat``
declare a column's data type, while ``known``, ``unknown``, and ``static``
describe when that column is available. For example, the v1 declarations above
map as follows:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - v1 argument
     - v2 argument
   * - ``time_idx``
     - ``time``
   * - ``target``
     - ``target``
   * - ``group_ids``
     - ``group``
   * - ``time_varying_known_reals``
     - ``num`` and ``known``
   * - ``time_varying_unknown_reals``
     - ``num`` and ``unknown``
   * - ``time_varying_known_categoricals``
     - ``cat`` and ``known``
   * - ``time_varying_unknown_categoricals``
     - ``cat`` and ``unknown``
   * - ``static_reals`` and ``static_categoricals``
     - ``num`` or ``cat``, plus ``static``
   * - ``max_encoder_length`` and ``max_prediction_length``
     - D2 DataModule arguments

The same column can appear in both a type list and a temporal-role list. The
v2 D1 layer only describes the columns; sequence construction, preprocessing,
and batching remain D2 responsibilities.

3. Move preprocessing and loaders to D2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a DataModule using the D1 dataset. This is where sequence lengths,
normalizers, categorical encoders, scalers, splits, and batch loading are
configured:

.. code-block:: python

   from pytorch_forecasting.data.data_module import (
       EncoderDecoderTimeSeriesDataModule,
   )

   datamodule = EncoderDecoderTimeSeriesDataModule(
       time_series_dataset=dataset,
       max_encoder_length=30,
       max_prediction_length=6,
       batch_size=64,
   )

Unlike v1, the DataModule is also the source of the metadata needed to
initialize a v2 model. The exact DataModule must match the input structure
expected by the model; see the compatibility information in
:doc:`v2 models <models_v2>`.

4. Choose the package or M-layer workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The P layer is the shortest migration path for a standard training workflow.
For example, a v2 TFT package separates configuration and accepts the D1
dataset directly:

.. code-block:: python

   from pytorch_forecasting.metrics import MAE, SMAPE
   from pytorch_forecasting.models.temporal_fusion_transformer import TFT_pkg_v2

   model_pkg = TFT_pkg_v2(
       datamodule_cfg={
           "max_encoder_length": 30,
           "max_prediction_length": 6,
           "batch_size": 64,
       },
       model_cfg={
           "loss": MAE(),
           "logging_metrics": [MAE(), SMAPE()],
           "hidden_size": 32,
           "attention_head_size": 1,
           "dropout": 0.1,
       },
       trainer_cfg={"max_epochs": 10, "accelerator": "auto"},
   )

   model_pkg.fit(dataset)
   predictions = model_pkg.predict(dataset)

``fit`` can also receive a pre-built D2 DataModule. ``predict`` accepts a D1
dataset, D2 DataModule, or PyTorch ``DataLoader``. For checkpoint
reproducibility, keep the configuration artifacts saved alongside the v2
checkpoint.

Use the M layer directly when you need full control over the Lightning
workflow. In that case, construct the DataModule first and pass its metadata to
the v2 model:

.. code-block:: python

   from lightning.pytorch import Trainer
   from pytorch_forecasting.metrics import MAE, SMAPE
   from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import TFT

   model = TFT(
       loss=MAE(),
       logging_metrics=[MAE(), SMAPE()],
       hidden_size=32,
       attention_head_size=1,
       dropout=0.1,
       metadata=datamodule.metadata,
   )

   Trainer(max_epochs=10, accelerator="auto").fit(model, datamodule=datamodule)

5. Validate before switching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare the v1 and v2 pipelines using the same:

* training and validation rows;
* encoder and prediction lengths;
* target scaling and categorical encodings;
* loss and evaluation metrics; and
* random seeds and trainer settings.

The output containers are not guaranteed to have the same structure. Inspect
the v2 prediction dictionary and adapt downstream code rather than assuming
that v1's ``output`` or decoded prediction fields are present.


Common replacements
-------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - v1 workflow
     - v2 workflow
   * - ``TimeSeriesDataSet``
     - ``TimeSeries`` plus a compatible D2 ``DataModule``
   * - ``TimeSeriesDataSet.from_dataset(...)``
     - Build the appropriate D1 data and D2 split/configuration explicitly
   * - ``dataset.to_dataloader(...)``
     - ``datamodule.train_dataloader()``, ``val_dataloader()``, or
       ``predict_dataloader()``
   * - ``Model.from_dataset(dataset, ...)``
     - Instantiate the v2 model with ``metadata=datamodule.metadata``
   * - ``Trainer.fit(model, train_dataloaders=..., ...)``
     - ``Trainer.fit(model, datamodule)`` or the package's ``fit()``
   * - ``model.predict(dataloader, ...)``
     - The v2 model's ``predict(dataloader, ...)`` or package's
       ``predict(dataset_or_datamodule, ...)``
   * - Dataset parameters saved manually
     - Package checkpoints can save model, DataModule, and metadata artifacts


New and migrated models
------------------------

The v2 model catalogue includes v2 implementations and package wrappers for
models such as TFT, DLinear, Samformer, TIDE, TimeXer, TSMixer, xLSTMTime,
DecoderMLP, SOFTS, SCINet, PatchTST, FreTS, ModernTCN, UniTS, and NHiTS.
Availability and DataModule compatibility can change during the beta period;
the authoritative list is the generated :doc:`v2 model overview <model_list>`
and the individual :doc:`M layer <m_layer_v2>` and
:doc:`P layer <pkg_v2>` references.

Having a v2 model with a similar name does not imply v1 checkpoint or output
compatibility. Treat each model as a new implementation and validate it against
your use case.


Migration checklist
-------------------

* [ ] Keep the current v1 environment and checkpoint available.
* [ ] Identify the v2 model and compatible D2 DataModule.
* [ ] Translate the v1 dataset column roles into a D1 ``TimeSeries``.
* [ ] Move preprocessing, scaling, encoding, splitting, and batching to D2.
* [ ] Choose the P layer for convenience or the M layer for custom Lightning
      control.
* [ ] Compare forecasts and metrics on an identical validation slice.
* [ ] Save and test v2 configuration artifacts with any v2 checkpoint.
* [ ] Keep v1 as the production fallback until v2 behavior is validated.


Further reading
---------------

* :doc:`API v2 overview <api_v2>`
* :doc:`v2 data documentation <data_v2>`
* :doc:`v2 models documentation <models_v2>`
* :doc:`v2 training tutorial <tutorials/ptf_V2_example>`
* `v2 development issue <https://github.com/sktime/pytorch-forecasting/issues/1736>`_
