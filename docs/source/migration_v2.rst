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

What stays the same
-------------------

What's new / not yet in v2
--------------------------

Getting help
------------
