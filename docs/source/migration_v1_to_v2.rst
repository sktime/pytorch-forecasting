Migrating models from v1 to v2
==============================

.. warning::
    The v2 model layer is in active development / beta. Use with caution.
    v1 remains stable for production — see :doc:`v1 API <api>`.

.. currentmodule:: pytorch_forecasting

.. note::
    This is a **developer** guide — how to migrate a model's *implementation* from
    the v1 API to the v2 four-layer architecture, per the roadmap goal *"migrate the
    models from v1 to v2 and deprecate v1"*
    (`#1993 <https://github.com/sktime/pytorch-forecasting/issues/1993>`_), aiming
    for **minimal changes to the model code**. To *use* v2 to build forecasts, see
    the v2 tutorials (``ptf_V2_example`` in :doc:`tutorials_v2`) instead; for the
    general contribution workflow, see the developer guide.

Overview
--------

Because the ``forward`` contract is unchanged between v1 and v2, migrating a model
is mostly a **re-organisation of the constructor and the surrounding package
plumbing**, not a rewrite of the network. A migrated model:

* inherits the v2 ``BaseModel`` (or ``TslibBaseModel`` for tslib models);
* takes its sizes from a ``metadata`` dict (supplied by the D2 DataModule) instead
  of from a dataset via ``from_dataset``;
* is split into a ``model.py`` (the network) and a ``model_pkg.py`` (the package
  class), and is registered so ``TestAllPtForecastersV2`` and ``check_estimator``
  cover it.

Changes to the model implementation
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - v1
     - v2
   * - Model base class
     - ``BaseModel`` (``models.base._base_model``)
     - ``BaseModel`` (``models.base._base_model_v2``); ``TslibBaseModel`` for tslib models
   * - Construction
     - ``@classmethod from_dataset(cls, dataset, ...)`` — sizes inferred from the dataset
     - ``__init__(..., metadata=None)`` — sizes read from the DataModule's ``metadata``
   * - Base ``super().__init__`` args
     - dataset-derived + hyperparameters
     - ``loss``, ``logging_metrics``, ``optimizer``, ``optimizer_params``, ``lr_scheduler``, ``lr_scheduler_params``
   * - Package class
     - inherits ``_BasePtForecaster``
     - inherits ``Base_pkg`` (adds ``get_cls`` / ``get_datamodule_cls`` / ``get_test_train_params``)
   * - Files
     - one class
     - ``model.py`` (network) + ``model_pkg.py`` (package / metadata)
   * - ``forward``
     - ``forward(x: dict) -> dict``
     - **unchanged**
   * - Test discovery
     - ``TestAllPtForecasters``
     - ``TestAllPtForecastersV2`` (via ``get_test_train_params`` + ``check_estimator``)

The data layer also changes (``TimeSeriesDataSet`` → a thin D1 ``TimeSeries`` plus a
D2 ``DataModule``), but a model **does not touch it directly** — it only consumes
the ``metadata`` the DataModule produces. See :doc:`data_v2`, :doc:`models_v2`,
:doc:`pkg_v2`.

Migration procedure
-------------------

**Step 1 — move the network into a v2** ``model.py``. Inherit the v2 ``BaseModel``;
the ``forward`` body usually transfers unchanged.

**Step 2 — replace** ``from_dataset`` **with** ``metadata``. In v1 the model read its
sizes from the dataset; in v2 they come from ``metadata`` (produced by the D2
DataModule) and are passed to ``__init__``:

.. code-block:: python

   # v1 — sizes inferred from the dataset via a factory classmethod
   @classmethod
   def from_dataset(cls, dataset, **kwargs):
       return super().from_dataset(dataset, **kwargs)

   # v2 — no from_dataset; sizes come from metadata (as TFT v2 does)
   import torch.nn as nn
   from pytorch_forecasting.models.base._base_model_v2 import BaseModel


   class MyModel(BaseModel):
       def __init__(
           self,
           loss,
           logging_metrics=None,
           optimizer="adam",
           optimizer_params=None,
           lr_scheduler=None,
           lr_scheduler_params=None,
           hidden_size=64,
           metadata=None,
       ):
           super().__init__(
               loss=loss,
               logging_metrics=logging_metrics,
               optimizer=optimizer,
               optimizer_params=optimizer_params,
               lr_scheduler=lr_scheduler,
               lr_scheduler_params=lr_scheduler_params,
           )
           self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
           self.metadata = metadata
           # read sizes from metadata and build layers, e.g.:
           enc_in = metadata["encoder_cont"] + metadata["encoder_cat"]
           self.encoder = nn.Linear(enc_in, hidden_size)

       @classmethod
       def _pkg(cls):
           from pytorch_forecasting.models.my_model._my_model_pkg import MyModel_pkg
           return MyModel_pkg

       def forward(self, x: dict) -> dict:
           ...  # unchanged from v1

For a real migrated model, see
``pytorch_forecasting/models/temporal_fusion_transformer/_tft_v2.py`` — it reads
``encoder_cont``, ``decoder_cont``, ``static_categorical_features``, etc. from
``metadata``.

**Step 3 — add a** ``model_pkg.py`` **package class** inheriting ``Base_pkg``, with
``_tags`` and the factory methods; point ``get_datamodule_cls`` at a compatible
DataModule and keep the first ``get_test_train_params`` entry ``{}`` (low-compute):

.. code-block:: python

   from pytorch_forecasting.base._base_pkg import Base_pkg


   class MyModel_pkg(Base_pkg):
       _tags = {"info:name": "MyModel", "authors": ["your-github-handle"]}

       @classmethod
       def get_cls(cls):
           from pytorch_forecasting.models.my_model._my_model import MyModel
           return MyModel

       @classmethod
       def get_datamodule_cls(cls):
           from pytorch_forecasting.data.data_module import (
               EncoderDecoderTimeSeriesDataModule,
           )
           return EncoderDecoderTimeSeriesDataModule

       @classmethod
       def get_test_train_params(cls):
           return [{}, {"hidden_size": 8}]

**Step 4 — register and check.** Register the package class so the ``all_objects``
registry and ``TestAllPtForecastersV2`` discover it, then validate the interface:

.. code-block:: python

   from pytorch_forecasting.utils._estimator_checks import check_estimator

   check_estimator(MyModel_pkg)

Migrating tslib models
----------------------

Models migrated from the Time-Series-Library inherit ``TslibBaseModel``
(``models.base._tslib_base_model_v2``) instead of ``BaseModel``. It handles the
tslib metadata (``context_length``, ``prediction_length``, ``feature_indices``,
``n_features``) and the shared initialisation boilerplate, so the subclass mostly
builds its layers from those. See ``TimeXer``
(``models/timexer/_timexer_v2.py``) and ``DLinear`` (``models/dlinear/_dlinear_v2.py``)
as references.

Unchanged components
--------------------

- ``forward(x: dict) -> dict`` — the network and its forward pass transfer directly.
- The PyTorch Lightning ``Trainer`` interface, and the loss / metric classes
  (``MAE``, ``SMAPE``, ``QuantileLoss``, …) from ``pytorch_forecasting.metrics``.

Migration status
----------------

Models already available in v2 (auto-generated from the registry — this list grows
as more are migrated):

.. model-overview-v2::

Remaining models to migrate, and v1 deprecation, are tracked in the roadmap
(`#1993 <https://github.com/sktime/pytorch-forecasting/issues/1993>`_) and the v2
work items (`#1974 <https://github.com/sktime/pytorch-forecasting/issues/1974>`_).

Getting help
------------

- Share feedback on the v2 rework in
  `issue #1736 <https://github.com/sktime/pytorch-forecasting/issues/1736>`_.
- Runnable examples: ``ptf_V2_example`` and ``tslib_v2_example`` in :doc:`tutorials_v2`.
