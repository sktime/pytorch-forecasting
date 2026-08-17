V2 Frequently Asked Questions
==============================

.. currentmodule:: pytorch_forecasting

This page covers common questions about the PyTorch Forecasting **v2 architecture**.
For questions about v1 (the legacy API), see :doc:`faq`.

For hands-on examples, consult the :ref:`Tutorials <tutorials>`.

.. contents:: Questions
   :local:
   :depth: 1

What is the architecture of v2?
-------------------------------

The architecture of PyTorch Forecasting v2 is composed of four distinct layers
to maximize flexibility and modularity:

* **Data Layer 1 (D1)** — :py:class:`~data.timeseries._timeseries_v2.TimeSeries`

  The D1 layer wraps a raw ``pandas.DataFrame`` and converts it into a
  ``torch.utils.data.Dataset``. It turns the raw tabular data into tensors
  (target ``y``, features ``x``, static features ``st``, time indices ``t``,
  and group identifiers) and creates a basic ``metadata`` dictionary that
  records column names, data types (numeric ``"F"`` / categorical ``"C"``),
  and future-known status (``"K"`` / ``"U"``). This metadata is later consumed
  by the D2 layer for model initialization.

  See the full API reference:
  :py:class:`~data.timeseries._timeseries_v2.TimeSeries`.

* **Data Layer 2 (D2)** — DataModule
  (e.g., :py:class:`~data.data_module.EncoderDecoderTimeSeriesDataModule`)

  The D2 layer is a ``LightningDataModule`` that takes a D1 dataset, applies
  sliding-window extraction with configurable encoder/decoder lengths, handles
  train/validation/test splits, preprocesses features (scaling, categorical
  encoding), and produces ``DataLoader`` instances for training and inference.
  It also exposes a ``metadata`` property that provides the shape information
  models need for initialization (e.g., number of encoder categorical features,
  number of targets, sequence lengths).

  See :ref:`Which data module should I use? <faq_v2_data_modules>` below for
  details on the available data modules.

* **Model Layer (M)** — PyTorch Lightning Module
  (e.g., :py:class:`~models.temporal_fusion_transformer._tft_v2.TFT`)

  The M layer is the core neural network. It is a standard
  ``LightningModule`` that focuses solely on the forward pass, loss
  calculation, and training/validation step logic. It receives batches
  produced by the D2 layer.

* **Package Layer (P)** — ``pkg`` wrapper
  (e.g., :py:class:`~models.temporal_fusion_transformer._tft_pkg_v2.TFT_pkg_v2`)

  The P layer is a high-level convenience wrapper. It manages
  model, datamodule, and trainer configurations in one place and exposes a
  simple ``fit()`` / ``predict()`` API. Using the ``pkg`` class is **not**
  the only way for users to interact with the library — you can always
  instantiate the model and datamodule manually and use the standard
  PyTorch Lightning ``Trainer`` workflow. The ``pkg`` layer simply makes
  common workflows easier. See :ref:`tutorials <tutorials>` for examples
  of both approaches.

  See the base class API:
  :py:class:`~base._base_pkg.Base_pkg`.


How to create the TimeSeries class?
-----------------------------------

You create a :py:class:`~data.timeseries._timeseries_v2.TimeSeries` object by
passing a ``pandas.DataFrame`` and specifying which columns represent time,
targets, groups, and features.

**Parameters of** ``TimeSeries``:

* ``data`` (``pd.DataFrame``) — The main data frame containing your time series
  observations. Column names must be strings.
* ``data_future`` (``pd.DataFrame``, optional) — An optional data frame with
  future-only data (e.g., known future covariates). May only contain columns
  that are in ``time``, ``group``, ``weight``, ``known``, or ``static``.
* ``time`` (``str``, optional) — Name of the integer-typed column representing
  the time index. Should increase by ``+1`` for consecutive observations.
  Defaults to the first column not in ``group``, ``weight``, ``target``, or
  ``static``.
* ``target`` (``str`` or ``List[str]``, optional) — Column name(s) for the
  forecasting target. Can be numerical or categorical. Defaults to the last
  column.
* ``group`` (``List[str]``, optional) — Column name(s) that uniquely identify
  a time series instance (e.g., ``["store_id", "product_id"]``). Together with
  ``time``, they uniquely identify each observation. If ``None``, a single
  series is assumed.
* ``weight`` (``str``, optional) — Column name for observation-level weights.
* ``num`` (``List[str]``, optional) — Numerical feature columns. Defaults to
  all columns with float/integer dtypes.
* ``cat`` (``List[str]``, optional) — Categorical feature columns. Defaults to
  all columns with object/boolean/category dtypes.
* ``known`` (``List[str]``, optional) — Variables that change over time and are
  **known** in the future (e.g., holidays, promotions). Defaults to all
  variables.
* ``unknown`` (``List[str]``, optional) — Variables that are **not known** in
  the future (e.g., weather observations). Defaults to no variables.
* ``static`` (``List[str]``, optional) — Variables that do **not** change over
  time (e.g., store location). Defaults to all variables not in ``known`` or
  ``unknown``.

**Example:**

.. code-block:: python

    from pytorch_forecasting.data.timeseries import TimeSeries

    dataset = TimeSeries(
        data=df,
        time="time_idx",
        target="target",
        group=["series_id"],
        num=["price", "temperature"],
        cat=["product_category"],
        known=["price", "product_category"],
        unknown=["temperature"],
        static=["store_region"],
    )

After construction, the dataset provides:

* ``len(dataset)`` — the number of time series groups.
* ``dataset[i]`` — a dict with tensors ``t``, ``y``, ``x``, ``group``,
  ``st``, and ``cutoff_time`` for the *i*-th group.
* ``dataset.get_metadata()`` — a dict with column names (``cols``), column
  types (``col_type``), and known/unknown status (``col_known``).

See the full API reference:
:py:class:`~data.timeseries._timeseries_v2.TimeSeries`.


What is the difference between TimeSeries, TimeSeriesDataSet (v1), and DataModule?
-----------------------------------------------------------------------------------

In **v1**, :py:class:`~data.timeseries.TimeSeriesDataSet` was a single
monolithic class responsible for everything: defining the dataset schema,
handling window extraction, batching, padding, scaling, categorical encoding,
and producing model-ready tensors. While convenient, this design tightly
coupled the data description with the batching strategy, making it difficult
to swap windowing approaches or reuse dataset definitions across different
model architectures.

In **v2**, this responsibility is **split** into two layers:

1. **TimeSeries (D1)** — :py:class:`~data.timeseries._timeseries_v2.TimeSeries`
   acts purely as a container for metadata and raw data. It describes *what*
   the data looks like (target, features, groups) and converts it to tensors,
   but does **not** handle windowing, batching, or splitting.

2. **DataModule (D2)** — classes like
   :py:class:`~data.data_module.EncoderDecoderTimeSeriesDataModule` handle
   *how* the data is processed: sliding-window creation, train/val/test splits,
   feature scaling, and ``DataLoader`` construction.

This separation means you can:

* Reuse a single ``TimeSeries`` definition with different data modules.
* Swap batching strategies without touching the dataset schema.
* Write new data modules for custom windowing or preprocessing logic.

For more details on the design philosophy, see the
`v2 architecture discussion <https://github.com/sktime/pytorch-forecasting/issues/1736>`_.


.. _faq_v2_data_modules:

Which data module should I use?
-------------------------------

PyTorch Forecasting v2 ships with multiple data modules in the
:py:mod:`~data.data_module` package. Each is designed for a different model
family:

* :py:class:`~data.data_module.EncoderDecoderTimeSeriesDataModule`
  — For traditional encoder-decoder models (e.g., TFT, NBeats, NHiTS).
  Uses ``max_encoder_length`` / ``max_prediction_length`` to create windows.

* :py:class:`~data.data_module.TslibDataModule`
  — For ``tslib``-style transformer architectures (e.g., Informer, Autoformer,
  TimeXer). Uses ``context_length`` / ``prediction_length`` and supports
  configurable ``window_stride`` for sliding-window control.

**When to use which:**

* If your model's ``forward()`` expects keys like ``encoder_cat``,
  ``encoder_cont``, ``decoder_cat``, ``decoder_cont``, use
  ``EncoderDecoderTimeSeriesDataModule``.
* If your model's ``forward()`` expects keys like ``history_cont``,
  ``history_cat``, ``future_cont``, ``future_cat``, use ``TslibDataModule``.

**When to implement a new data module:**

Only create a new data module when your model requires:

* Unique data structures or non-standard time-series features that existing
  modules cannot parse.
* Custom metadata preparation (``_prepare_metadata``) to configure model
  architecture shapes.
* Specialized sample-level preprocessing (``_preprocess_data``) or custom
  batch-collating logic.

See the `data module extension template
<https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v2/data_module>`_
for a ready-to-copy starting point.

**Key parameters of** ``EncoderDecoderTimeSeriesDataModule``:

* ``time_series_dataset`` — A ``TimeSeries`` (D1) instance.
* ``max_encoder_length`` (int, default=30) — Max encoder input length.
* ``min_encoder_length`` (int, optional) — Min encoder length. Defaults to
  ``max_encoder_length``.
* ``max_prediction_length`` (int, default=1) — Max decoder output length.
* ``min_prediction_length`` (int, optional) — Min decoder length. Defaults
  to ``max_prediction_length``.
* ``allow_missing_timesteps`` (bool, default=False) — Allow gaps in time.
* ``add_relative_time_idx`` (bool, default=False) — Add relative time index
  feature.
* ``add_target_scales`` (bool, default=False) — Add target scaling info.
* ``target_normalizer`` (default="auto") — Normalizer for the target. If
  ``"auto"``, uses ``RobustScaler``.
* ``categorical_encoders`` (dict, optional) — Custom encoders per column.
* ``scalers`` (dict, optional) — Custom scalers per column.
* ``batch_size`` (int, default=32) — Batch size for DataLoaders.
* ``num_workers`` (int, default=0) — Number of DataLoader workers.
* ``train_val_test_split`` (tuple, default=(0.7, 0.15, 0.15)) — Split ratios.

**Key parameters of** ``TslibDataModule``:

* ``time_series_dataset`` — A ``TimeSeries`` (D1) instance.
* ``context_length`` (int) — Length of the input context window.
* ``prediction_length`` (int) — Length of the prediction window.
* ``freq`` (str, default="h") — Frequency of the time series.
* ``add_relative_time_idx`` (bool, default=False) — Add relative time index.
* ``add_target_scales`` (bool, default=False) — Add target scaling info.
* ``target_normalizer`` (default="auto") — Normalizer for the target.
* ``scalers`` (dict, optional) — Custom scalers per column.
* ``shuffle`` (bool, default=True) — Shuffle data each epoch.
* ``window_stride`` (int, default=1) — Stride for the sliding window.
* ``batch_size`` (int, default=32) — Batch size for DataLoaders.
* ``num_workers`` (int, default=0) — Number of DataLoader workers.
* ``train_val_test_split`` (tuple, default=(0.7, 0.15, 0.15)) — Split ratios.

For the full API reference, see:

* :py:class:`~data.data_module.EncoderDecoderTimeSeriesDataModule`
* :py:class:`~data.data_module.TslibDataModule`

**Example:**

.. code-block:: python

    from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule

    data_module = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=dataset,
        max_encoder_length=24,
        max_prediction_length=12,
        batch_size=64,
    )


How to use pkg classes to perform fit and predict?
--------------------------------------------------

The ``pkg`` classes (P layer) provide a **convenient**, high-level wrapper
that bundles model, datamodule, and trainer configuration. They inherit from
:py:class:`~base._base_pkg.Base_pkg`.

.. important::

   Using ``pkg`` is **not** the only way to train and predict with
   PyTorch Forecasting v2. You can always create the model and data module
   manually and use the standard ``lightning.Trainer`` directly. The ``pkg``
   layer is a convenience shortcut. For examples of the manual approach, see
   the :ref:`Tutorials <tutorials>`.

**Features of the** ``pkg`` **class:**

* Accepts ``model_cfg``, ``datamodule_cfg``, and ``trainer_cfg`` as
  simple dictionaries (or paths to ``.yaml`` / ``.pkl`` files).
* Automatically builds the model and data module from config.
* Provides ``fit(data)`` that accepts either a ``TimeSeries`` (D1) or
  ``LightningDataModule`` (D2) directly.
* Provides ``predict(data)`` that accepts ``TimeSeries``, ``DataModule``,
  or raw ``DataLoader``.
* Handles checkpoint saving and loading, including automatic serialization
  of ``model_cfg``, ``datamodule_cfg``, and ``metadata`` alongside the
  checkpoint.
* Supports loading a trained model from a checkpoint via ``ckpt_path``.

**Parameters of** ``Base_pkg``:

* ``model_cfg`` (dict, optional) — Model initialization parameters.
* ``trainer_cfg`` (dict, optional) — ``lightning.Trainer`` parameters.
* ``datamodule_cfg`` (dict or path, optional) — Data module parameters.
  Can be a dict, or a path to a ``.pkl`` / ``.yaml`` file for
  reproducible inference.
* ``ckpt_path`` (str or Path, optional) — Path to a saved checkpoint
  to resume from.

**Example:**

.. code-block:: python

    from pytorch_forecasting.models.temporal_fusion_transformer import TFT_pkg_v2

    model_pkg = TFT_pkg_v2(
        model_cfg={"hidden_size": 32},
        datamodule_cfg={"max_encoder_length": 24, "max_prediction_length": 12},
        trainer_cfg={"max_epochs": 10},
    )

    # fit directly on a TimeSeries (D1) — pkg builds the datamodule for you
    best_ckpt = model_pkg.fit(dataset)

    # predict — pass TimeSeries, DataModule, or DataLoader
    predictions = model_pkg.predict(dataset)

**Equivalent manual workflow (without pkg):**

.. code-block:: python

    from lightning import Trainer
    from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
    from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import TFT

    data_module = EncoderDecoderTimeSeriesDataModule(
        dataset, max_encoder_length=24, max_prediction_length=12
    )
    data_module.setup("fit")

    model = TFT(metadata=data_module.metadata, hidden_size=32)
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, datamodule=data_module)

See the full API reference: :py:class:`~base._base_pkg.Base_pkg`.


How to add a new model to the package in v2?
--------------------------------------------

Adding a new model involves two main steps. Ready-to-copy templates are
available in the `extension templates directory
<https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v2>`_.

**Step 1 — Create the Model Layer (M):** ``model.py``

Inherit from ``BaseModel`` and implement:

* ``__init__()``: Initialize network components. Must call
  ``self.save_hyperparameters()`` and ``super().__init__()``.
* ``_pkg()``: A ``@classmethod`` that imports and returns your ``pkg`` class.
* ``forward(x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]``:
  The PyTorch forward pass.

See the `model extension template
<https://github.com/sktime/pytorch-forecasting/blob/main/extension_templates/v2/model_simple/model.py>`_.

**Step 2 — Create the Package Layer (P):** ``model_pkg.py``

Inherit from :py:class:`~base._base_pkg.Base_pkg` and implement:

* ``get_cls()``: Import and return your model class.
* ``get_datamodule_cls()``: Import and return the compatible data module
  class. Inspect the available modules in
  `pytorch_forecasting.data.data_module
  <https://github.com/sktime/pytorch-forecasting/tree/main/pytorch_forecasting/data/data_module>`_
  to find one whose output tensor keys match your model's ``forward()``
  signature. If none exist, implement a custom data module (see the
  `data module extension template
  <https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v2/data_module>`_).
* ``get_test_train_params()``: Return a list of dicts for CI testing.
  The first element must be ``{}`` to test defaults.
* ``_tags``: A dictionary defining your model's metadata and capabilities:

  - ``info:name`` — Human-readable model name.
  - ``info:pred_type`` — e.g., ``["point"]``, ``["quantile"]``.
  - ``info:y_type`` — e.g., ``["numeric"]``.
  - ``info:compute`` — Compute intensity (1–5).
  - ``authors`` — List of GitHub usernames.
  - ``capability:exogenous`` — Whether model supports exogenous variables.
  - ``capability:multivariate`` — Whether model supports multivariate targets.
  - ``capability:pred_int`` — Whether model supports prediction intervals.
  - ``capability:flexible_history_length`` — Variable-length history support.
  - ``capability:cold_start`` — Predictions with little/no history.

See the `package extension template
<https://github.com/sktime/pytorch-forecasting/blob/main/extension_templates/v2/model_simple/model_pkg.py>`_.

**Validate your implementation:**

.. code-block:: python

    from pytorch_forecasting.utils._estimator_checks import check_estimator
    from pytorch_forecasting.models.my_model import MyModel

    check_estimator(MyModel)
