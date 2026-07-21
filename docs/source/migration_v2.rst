Migrating from v1 to v2
=======================

.. warning::
    The v2 modules are in active development / beta. Use with caution.
    v1 remains stable for production pipelines — see :doc:`v1 API <api>`.

.. currentmodule:: pytorch_forecasting

.. note::
    v1 is still supported and is **not** deprecated. This guide is for existing
    v1 users moving code to v2. v2 is evolving and its API may still change —
    track progress in `issue #1974 <https://github.com/sktime/pytorch-forecasting/issues/1974>`_.

High-level changes at a glance
------------------------------

Concept mapping: v1 → v2
------------------------

In v1, a single ``TimeSeriesDataSet`` configured data handling *and* fed the model
via ``Model.from_dataset(...)``. In v2 those responsibilities split across layers: a thin
``TimeSeries`` dataset (:doc:`data_v2`), a ``DataModule`` that does preprocessing and
batching, a data-agnostic model, and an optional package wrapper (:doc:`pkg_v2`). The
table maps the pieces you touch when porting code.

.. list-table::
   :header-rows: 1
   :widths: 34 44 22

   * - v1
     - v2
     - Notes
   * - ``TimeSeriesDataSet(data, ...)``
     - ``TimeSeries(data=..., ...)`` (D1) + a ``DataModule`` (D2)
     - Dataset is now thin; preprocessing/batching move to the DataModule.
   * - ``time_idx="..."``
     - ``time="..."``
     - Renamed.
   * - ``group_ids=[...]``
     - ``group=[...]``
     - Renamed.
   * - ``static_reals`` / ``static_categoricals``
     - ``static=[...]`` (+ list each column in ``num=`` / ``cat=``)
     - v2 declares role (``static`` / ``known`` / ``unknown``) and dtype (``num`` / ``cat``) separately.
   * - ``time_varying_known_reals`` / ``..._known_categoricals``
     - ``known=[...]``
     - Role = known-in-future.
   * - ``time_varying_unknown_reals`` / ``..._unknown_categoricals``
     - ``unknown=[...]``
     - Role = unknown-in-future.
   * - ``target_normalizer=`` / ``categorical_encoders`` / ``scalers`` on the dataset
     - the same args on the ``DataModule`` (or ``datamodule_cfg``)
     - Preprocessing config moved from dataset to DataModule.
   * - ``training.to_dataloader(...)`` + ``TimeSeriesDataSet.from_dataset(..., predict=True)``
     - the ``DataModule`` builds train/val/test dataloaders
     - No manual dataloader/validation-set construction.
   * - ``Model.from_dataset(training, ...)``
     - ``TFT(..., metadata=data_module.metadata)`` **or** ``TFT_pkg_v2(model_cfg=...)``
     - Model no longer built from the dataset; it receives metadata from the DataModule.
   * - ``trainer.fit(model, train_dl, val_dl)``
     - ``trainer.fit(model, data_module)`` **or** ``pkg.fit(dataset)``
     - Lightning ``Trainer`` unchanged; package wrapper offers a one-call ``fit``.
   * - ``model.predict(dataloader)``
     - ``pkg.predict(dataset, return_info=[...])``
     - Package exposes a unified ``predict``.
   * - ``QuantileLoss`` / ``MAE`` / ``SMAPE`` (``pytorch_forecasting.metrics``)
     - unchanged
     - v2 reuses the v1 metrics suite (see :doc:`api_v2`).

End-to-end example: TFT
-----------------------

The same TFT workflow — load data, build the model, train (and predict) — in
each API, using the bundled toy dataset (``load_toydata``).

**v1**

.. code-block:: python

   import lightning.pytorch as pl

   from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
   from pytorch_forecasting.data import TorchNormalizer
   from pytorch_forecasting.data.examples import load_toydata
   from pytorch_forecasting.metrics import MAE

   data = load_toydata(num_series=100, seq_length=50)
   data["category"] = data["category"].astype(str)
   data["static_feature_cat"] = data["static_feature_cat"].astype(str)

   cutoff = data["time_idx"].max() - 1
   training = TimeSeriesDataSet(
       data[lambda x: x.time_idx <= cutoff],
       time_idx="time_idx",
       target="y",
       group_ids=["series_id"],
       max_encoder_length=30,
       max_prediction_length=1,
       static_reals=["static_feature"],
       static_categoricals=["static_feature_cat"],
       time_varying_known_reals=["future_known_feature"],
       time_varying_unknown_reals=["x"],
       time_varying_unknown_categoricals=["category"],
       target_normalizer=TorchNormalizer(),
   )
   train_dataloader = training.to_dataloader(train=True, batch_size=32)

   tft = TemporalFusionTransformer.from_dataset(
       training,
       hidden_size=64,
       attention_head_size=4,
       dropout=0.1,
       hidden_continuous_size=16,
       loss=MAE(),
       optimizer="adam",
   )
   pl.Trainer(max_epochs=5, accelerator="cpu").fit(
       tft, train_dataloaders=train_dataloader
   )

**v2 (recommended: package wrapper)**

The dataset is the thin D1 ``TimeSeries``; preprocessing (encoders, scalers,
target normalizer) and batching move into the DataModule config the package
builds for you.

.. code-block:: python

   from sklearn.preprocessing import StandardScaler

   from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer
   from pytorch_forecasting.data.examples import load_toydata
   from pytorch_forecasting.data.timeseries import TimeSeries
   from pytorch_forecasting.metrics import MAE, SMAPE
   from pytorch_forecasting.models.temporal_fusion_transformer._tft_pkg_v2 import (
       TFT_pkg_v2,
   )

   data = load_toydata(num_series=100, seq_length=50)

   dataset = TimeSeries(
       data=data,
       time="time_idx",
       target="y",
       group=["series_id"],
       num=["x", "future_known_feature", "static_feature"],
       cat=["category", "static_feature_cat"],
       known=["future_known_feature"],
       unknown=["x", "category"],
       static=["static_feature", "static_feature_cat"],
   )

   pkg = TFT_pkg_v2(
       model_cfg=dict(
           loss=MAE(),
           logging_metrics=[MAE(), SMAPE()],
           optimizer="adam",
           hidden_size=64,
           num_layers=2,
           attention_head_size=4,
           dropout=0.1,
       ),
       trainer_cfg=dict(max_epochs=5, accelerator="cpu"),
       datamodule_cfg=dict(
           max_encoder_length=30,
           max_prediction_length=1,
           batch_size=32,
           categorical_encoders={
               "category": NaNLabelEncoder(add_nan=True),
               "static_feature_cat": NaNLabelEncoder(add_nan=True),
           },
           scalers={
               "x": StandardScaler(),
               "future_known_feature": StandardScaler(),
               "static_feature": StandardScaler(),
           },
           target_normalizer=TorchNormalizer(),
       ),
   )
   pkg.fit(dataset)
   preds = pkg.predict(dataset, return_info=["index", "y"])

**v2 (full control: explicit layers)**

If you want the layers explicit — the same ``dataset`` above, a DataModule you
build yourself, a model that receives its ``metadata``, and a plain Lightning
``Trainer``:

.. code-block:: python

   from lightning.pytorch import Trainer
   from sklearn.preprocessing import StandardScaler

   from pytorch_forecasting.data.data_module import (
       EncoderDecoderTimeSeriesDataModule,
   )
   from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer
   from pytorch_forecasting.metrics import MAE, SMAPE
   from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import TFT

   data_module = EncoderDecoderTimeSeriesDataModule(
       time_series_dataset=dataset,
       max_encoder_length=30,
       max_prediction_length=1,
       batch_size=32,
       categorical_encoders={
           "category": NaNLabelEncoder(add_nan=True),
           "static_feature_cat": NaNLabelEncoder(add_nan=True),
       },
       scalers={
           "x": StandardScaler(),
           "future_known_feature": StandardScaler(),
           "static_feature": StandardScaler(),
       },
       target_normalizer=TorchNormalizer(),
   )
   model = TFT(
       loss=MAE(),
       logging_metrics=[MAE(), SMAPE()],
       optimizer="adam",
       hidden_size=64,
       num_layers=2,
       attention_head_size=4,
       dropout=0.1,
       metadata=data_module.metadata,
   )
   Trainer(max_epochs=5, accelerator="cpu").fit(model, data_module)

For a runnable end-to-end notebook, see :doc:`tutorials_v2` (``ptf_V2_example``).

What stays the same
-------------------

What's new / not yet in v2
--------------------------

Getting help
------------
