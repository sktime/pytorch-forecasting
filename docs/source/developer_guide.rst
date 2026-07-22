.. _developer_guide:

Developer Guide
===============

This guide is for people who want to **develop** ``pytorch-forecasting`` — to
contribute code, not just use the library. It covers both **v1** (stable, still
maintained) and **v2** (the active redesign), and is organised around the
architecture and the v1 → v2 transition, since that is where most current
contribution happens.

``pytorch-forecasting`` is part of the `sktime <https://www.sktime.net>`_
ecosystem. Project-wide conventions that are not specific to this package —
coding standards, the git and reviewer workflow, enhancement proposals,
deprecation policy, and governance — follow the sktime developer documentation
and are linked rather than repeated here:

* sktime developer guide: https://www.sktime.net/en/stable/developer_guide.html
* sktime contributing guide: https://www.sktime.net/en/latest/get_involved/contributing.html

.. note::

   Using the library to build forecasts (rather than changing its source)? See
   the :doc:`installation guide <installation>`, and — if you are migrating your
   own v1 forecasting *code* to the v2 API — the migration guide in the
   :doc:`v2 documentation <api_v2>`. This developer guide is about changing the
   library itself.

Overview
--------

A contribution to ``pytorch-forecasting`` is usually a new **estimator** — a
model, a data module, or a metric — most often for the v2 API. Whatever you add
follows a small set of shared conventions and, where one exists, an official
extension template. This guide walks through setting up your environment, the
architecture (so the conventions make sense), and how to add, verify, and submit
a contribution.

Setup / CI
----------

Setup
~~~~~

Follow :doc:`installation` (see *Contributing to pytorch-forecasting*) to create
a fork, clone it, and set up an editable virtual environment with the developer
dependencies — ``pip install -e ".[dev]"`` (add ``",all_extras"`` for the soft
dependencies used by some estimators). Then install the pre-commit hooks:

.. code-block:: bash

   pre-commit install

The ``[dev]`` extra installs the linters, ``pytest`` (+ ``pytest-xdist`` /
``pytest-cov`` / ``pytest-sugar``), and the docs build stack.

Code quality
~~~~~~~~~~~~

Pre-commit is the single entry point for code style. After ``pre-commit install``
the following hooks run automatically on every ``git commit``:

* **trailing-whitespace** — strips trailing whitespace.
* **end-of-file-fixer** — ensures files end with a newline.
* **check-yaml** — validates YAML syntax.
* **check-ast** — validates Python syntax.
* **ruff** (``--fix``) — linting with auto-fix; runs the rule sets described
  below.
* **ruff-format** — opinionated formatter (line length 88, double quotes, space
  indent).
* **nbqa-black** — applies Black formatting to Jupyter notebooks.
* **nbqa-ruff** — applies ruff linting to Jupyter notebooks.
* **nbqa-check-ast** — validates notebook cell syntax.

To run all hooks manually (useful before opening a PR):

.. code-block:: bash

   pre-commit run --all-files

To invoke ruff directly:

.. code-block:: bash

   ruff check .          # lint only
   ruff format .         # format only

**Ruff rule sets** (from ``[tool.ruff.lint]`` in ``pyproject.toml``):

* ``select``: ``E``, ``F``, ``W`` (pycodestyle/pyflakes errors and warnings),
  ``C4`` (flake8-comprehensions), ``S`` (flake8-bandit security).
* ``extend-select``: ``I`` (isort import ordering), ``UP`` (pyupgrade modern
  syntax).

**Docstrings:** all public classes and functions must use NumPy-style
docstrings. For the full coding standards — naming conventions, type
annotations, and import organisation — follow the
`sktime coding standards <https://www.sktime.net/en/stable/developer_guide/coding_standards.html>`_.

Test
~~~~

Tests live in two directories (configured in ``[tool.pytest.ini_options]``):
``tests/`` and ``pytorch_forecasting/tests/``. The default ``addopts`` in
``pyproject.toml`` enables coverage reporting automatically:

.. code-block:: bash

   pytest                              # full suite with coverage
   pytest -n auto                      # parallel execution via pytest-xdist
   pytest -k "tft"                     # filter by name
   pytest --cov=pytorch_forecasting \
          --cov-report=html            # HTML report written to htmlcov/

Coverage output lands in ``htmlcov/``; the terminal report shows missing lines
per file (``--cov-report=term-missing:skip-covered``).

**Registry auto-discovery.** Both the v1 registry (``TestAllPtForecasters``)
and the v2 registry (``TestAllPtForecastersV2``) discover estimators through
``all_objects``. A correctly registered model is therefore picked up and tested
automatically — you rarely need to write a per-model test file. What you do
need to provide is ``get_test_train_params`` on the package class: keep every
configuration small (minimal sequence length, tiny hidden sizes) so the
registry-driven tests stay fast in CI.

Continuous integration
~~~~~~~~~~~~~~~~~~~~~~

On every push or pull-request targeting ``main``,
``.github/workflows/test.yml`` runs five jobs:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Job
     - What it does
   * - **code-quality**
     - Runs ``pre-commit`` on the changed files only. All subsequent jobs
       depend on this one passing.
   * - **no-softdeps**
     - Runs the full ``pytest`` suite *without* soft dependencies across all
       supported Python versions (3.10–3.14) on Linux, macOS, and Windows.
       Ensures the core package works without optional extras.
   * - **pytest**
     - Runs ``pytest`` with all extras across the same Python × OS matrix.
       Coverage is uploaded to Codecov. Depends on **no-softdeps** passing.
   * - **dependency snapshot**
     - Installs the package pinned to a recent dependency snapshot and runs
       ``pytest``, guarding against regressions from dependency updates.
   * - **notebook tutorials**
     - Executes the example notebooks to catch regressions.

All jobs must pass before a pull request can be merged.

Extension library
~~~~~~~~~~~~~~~~~

The ``extension_templates/`` directory holds the official copy-and-fill
templates for every recognised contribution type (v2 model, v2 data module,
metric, v1 network). See `Adding a contribution`_ for details and code
skeletons.

Architecture
------------

v1 architecture
~~~~~~~~~~~~~~~

In v1 a single ``TimeSeriesDataSet`` handles ingestion, preprocessing, and
batching in one class. Models inherit ``BaseModel`` (in
``pytorch_forecasting.models.base._base_model``), which mixes in several
capabilities — training step, logging, prediction helpers — and are
constructed with ``Model.from_dataset(dataset, ...)``; the model reads its
input sizes, embedding dimensions, and categorical vocabulary directly from
the dataset object at instantiation. A lightweight package class inheriting
``_BasePtForecaster`` carries tags and metadata and participates in the
registry (``TestAllPtForecasters``). The tight coupling between the dataset
object and the model constructor is what the v2 redesign addresses.

.. mermaid::

   flowchart TD
       raw["Raw data (pandas DataFrame)"] --> tsds["TimeSeriesDataSet<br/>ingest + preprocess + batch"]
       tsds -->|from_dataset| model["BaseModel<br/>network + forward + training"]
       model --- pkg["_BasePtForecaster<br/>tags + metadata"]

v2 architecture
~~~~~~~~~~~~~~~

v2 redesigns the API around three goals:

* **Decoupled models and data** — a model no longer depends on a specific
  data-handling class, so it works with plain tensors and dataloaders.
* **A unified, exchangeable-model API** — a high-level ``fit`` / ``predict``
  wrapper lets you swap architectures without changing the workflow.
* **Easier extensibility** — clear layer boundaries, with a template per component.

These goals are realised as a strict **four-layer** design that separates data
ingestion, preprocessing, the forecasting algorithm, and high-level
orchestration:

* **D1 — Dataset** (``TimeSeries``): converts raw tabular data (pandas
  DataFrames) into ``torch`` tensors and extracts base metadata. It performs no
  preprocessing or batching, keeping it lightweight and modular. See
  :doc:`data_v2`.
* **D2 — DataModule** (a ``LightningDataModule``): sits on top of D1, does the
  preprocessing and batching, and assembles the ``metadata`` a model needs at
  initialisation. Concrete implementations include
  ``EncoderDecoderTimeSeriesDataModule`` and ``TslibDataModule``; the
  model-to-DataModule compatibility table is in :doc:`models_v2`. See
  :doc:`data_v2`.
* **Model** (``BaseModel`` in ``...base._base_model_v2`` — a **different class**
  from v1's same-named ``BaseModel``): the forecasting algorithm as a PyTorch
  Lightning module, agnostic to the data pipeline and initialised from explicit
  hyperparameters plus the ``metadata`` from D2. ``forward(x: dict) -> dict`` is
  unchanged from v1. Models ported from tslib inherit ``TslibBaseModel`` instead.
  See :doc:`models_v2`.
* **Package** (``Base_pkg``): a thin wrapper that orchestrates the layers,
  exposes the unified ``fit`` / ``predict`` interface, declares DataModule
  compatibility via ``get_datamodule_cls()``, and houses the test fixtures. See
  :doc:`pkg_v2`.

.. mermaid::

   flowchart TD
       raw["Raw data (pandas DataFrame)"] -->|raw| d1["D1: TimeSeries"]
       d1 -->|"tensors + base metadata"| d2["D2: LightningDataModule"]
       d2 -->|"metadata + dataloaders"| model["Model: BaseModel<br/>(LightningModule)"]
       user(["User"]) ==>|"high-level API:<br/>fit / predict"| pkg["Package: Base_pkg"]
       pkg -.->|orchestrates| d1
       user -->|"low-level API:<br/>drive the 3 stages via Lightning"| d1

**Two API levels:**

* **High-level API** — the Package's ``fit`` / ``predict``, which wires the
  ``D1 → D2 → Model`` stages together for you.
* **Low-level API** — drive the same **three-stage pipeline**
  (``TimeSeries`` → ``LightningDataModule`` → ``BaseModel``) directly through a
  PyTorch Lightning ``Trainer``.

Nothing is hidden — the Package is a convenience wrapper over the three-stage
pipeline. Detailed references for each layer are in :doc:`data_v2`,
:doc:`models_v2`, :doc:`pkg_v2`, and :doc:`api_v2`; for runnable end-to-end
examples, see the v2 tutorials :doc:`tutorials/ptf_V2_example` and
:doc:`tutorials/tslib_v2_example`.

.. warning::

   The v2 modules are under active development and are in beta. Use the v2 API
   with caution; it is not yet production-ready. The stable v1 API remains
   available for production pipelines. Feedback is welcome in
   `issue #1736 <https://github.com/sktime/pytorch-forecasting/issues/1736>`_.

v1 to v2
~~~~~~~~

The table below summarises the shifts that matter for contributors: the data
layer splits into D1/D2, ``from_dataset`` gives way to ``metadata``, the package
base gains ``get_datamodule_cls()``, and the ``forward`` contract is unchanged
(so migrating a model is mostly re-organisation, not a rewrite).

.. list-table:: v1 → v2 at a glance
   :header-rows: 1
   :widths: 18 24 24 34

   * - Aspect
     - v1
     - v2
     - What changed & why
   * - Data handling
     - one ``TimeSeriesDataSet`` (ingest + preprocess + batch)
     - ``TimeSeries`` (D1) + ``LightningDataModule`` (D2)
     - split into layers so models decouple from data handling and can work with
       plain tensors and dataloaders
   * - Model construction
     - ``Model.from_dataset(dataset)``
     - ``Model.__init__(metadata, ...)`` — no ``from_dataset``
     - the model no longer reads a dataset object; the D2 DataModule supplies a
       plain ``metadata`` dict, making models modular and independently testable
   * - Model base class
     - ``BaseModel`` (``_base_model``)
     - ``BaseModel`` (``_base_model_v2``), or ``TslibBaseModel`` for tslib models
     - same class name, different module — do not confuse the two
   * - ``forward`` contract
     - ``forward(x: dict) -> dict``
     - ``forward(x: dict) -> dict``
     - **unchanged** — this is why migrating is structural re-organisation, not a
       rewrite
   * - Package base
     - ``_BasePtForecaster``
     - ``Base_pkg`` (extends ``_BasePtForecasterV2``) + ``get_datamodule_cls()``
     - the package now explicitly declares DataModule compatibility
   * - API surface
     - Lightning module used directly
     - high-level ``fit`` / ``predict`` via ``Base_pkg`` + low-level 3-stage
       Lightning pipeline
     - adds a unified workflow without hiding the Lightning interface
   * - Test discovery
     - ``TestAllPtForecasters``
     - ``TestAllPtForecastersV2``
     - registry-based auto-discovery via ``get_test_train_params``
   * - Status
     - stable, production-ready
     - experimental (beta) — API may change
     - use v1 for production; v2 for new development and testing

Adding a contribution
---------------------

A contribution is usually a new estimator — a model, a data module, or a metric.
Every v2 estimator follows the same conventions (below); then, for each kind,
there is an official template to copy and fill in.

Conventions every contribution follows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Package class inheriting** ``Base_pkg`` **with a** ``_tags`` **dictionary**
  describing the estimator's name, compute cost, prediction type, target type,
  authors, capabilities, and soft dependencies.
* **Registered for auto-discovery** so the ``all_objects`` registry and the
  ``TestAllPtForecastersV2`` test suite find it automatically — no per-estimator
  test file needed. (v2 estimators carry the ``object_type`` tag
  ``forecaster_pytorch_v2``, distinguishing them from v1's ``forecaster_pytorch_v1``.)
* **Passes** ``check_estimator`` (run it before opening a PR; see below).
* **Provides** ``get_test_train_params`` — the first entry must be ``{}`` so
  default construction is tested, and every entry must be low-compute so CI does
  not time out.
* **Guarded soft-dependency imports** — import any non-core package *inside the
  method that needs it*, not at module top level, so the core still imports.

The ``_tags`` dictionary is how the framework understands your estimator (keys
from the template at ``extension_templates/v2/model_simple/model_pkg.py``):

.. code-block:: python

   _tags = {
       # Model name — MUST match the model class name exactly.
       "info:name": "MyModel",
       # Approximate compute cost: 1 = lightweight (MLP) .. 5 = very heavy.
       "info:compute": 2,
       # Prediction output type(s): "point" / "quantile" / "distr".
       "info:pred_type": ["point"],
       # Target type(s) supported: "numeric" / "category".
       "info:y_type": ["numeric"],
       # GitHub handles of the contributors.
       "authors": ["your-github-handle"],
       # Capability flags.
       "capability:exogenous": True,               # uses exogenous covariates?
       "capability:multivariate": True,            # multivariate target?
       "capability:pred_int": False,               # prediction intervals?
       "capability:flexible_history_length": True, # variable-length encoder?
       "capability:cold_start": False,             # works without long history?
       # Soft dependencies required to run this model (empty if none).
       "python_dependencies": [],
   }

Validate interface compatibility before opening a PR:

.. code-block:: python

   from pytorch_forecasting.utils._estimator_checks import check_estimator

   check_estimator(MyModel_pkg)

Official templates
~~~~~~~~~~~~~~~~~~

For each kind there is an official template in ``extension_templates/`` — copy
it, rename it, and work through the ``todo`` comments. Templates are scaffolds,
not base classes to import.

**Add a new v2 model** (``extension_templates/v2/model_simple/``) — two files.
``model.py`` holds the network (mandatory: ``_pkg`` and ``forward``;
``save_hyperparameters`` before ``super().__init__``):

.. code-block:: python

   import torch
   from pytorch_forecasting.models.base._base_model_v2 import BaseModel


   class MyModel(BaseModel):
       def __init__(self, loss, hidden_size=64, metadata=None, **kwargs):
           self.save_hyperparameters(ignore=["loss", "logging_metrics", "optimizer"])
           super().__init__(loss=loss, **kwargs)
           # metadata (from the DataModule) carries variable counts / sizes;
           # use it here to build your layers, e.g. torch.nn.Linear(...)

       @classmethod
       def _pkg(cls):
           from pytorch_forecasting.models.my_model.model_pkg import MyModel_pkg
           return MyModel_pkg

       def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
           # todo: read tensors from x, return e.g. {"prediction": y_hat}
           ...

The v2 ``BaseModel`` constructor accepts ``loss``, ``logging_metrics``,
``optimizer``, ``optimizer_params``, ``lr_scheduler``, and
``lr_scheduler_params`` (pass them through ``super().__init__()``); your model
additionally takes ``metadata`` from the DataModule to size its layers. **For a
model ported from tslib, inherit** ``TslibBaseModel``
(``pytorch_forecasting.models.base._tslib_base_model_v2``) instead of
``BaseModel`` — it handles the metadata and initialisation boilerplate shared by
tslib models (e.g. ``TimeXer``, ``DLinear``).

``model_pkg.py`` holds the metadata and factory methods (mandatory: ``get_cls``,
``get_datamodule_cls``, ``get_test_train_params``):

.. code-block:: python

   from pytorch_forecasting.base._base_pkg import Base_pkg


   class MyModel_pkg(Base_pkg):
       _tags = {"info:name": "MyModel", "authors": ["your-github-handle"], ...}

       @classmethod
       def get_cls(cls):
           from pytorch_forecasting.models.my_model.model import MyModel
           return MyModel

       @classmethod
       def get_datamodule_cls(cls):
           from pytorch_forecasting.data.data_module import (
               EncoderDecoderTimeSeriesDataModule,
           )
           return EncoderDecoderTimeSeriesDataModule

       @classmethod
       def get_test_train_params(cls):
           return [{}, {"paramb": "other"}]   # first {} tests defaults; keep small

**Add a new v2 data module** (``extension_templates/v2/data_module/``) — a D2
``LightningDataModule`` (mandatory: ``_prepare_metadata``, the ``metadata``
property, ``_preprocess_data``, ``setup``) plus a private ``Dataset``
(mandatory: ``__getitem__``, returning the item dict the model's ``forward``
expects):

.. code-block:: python

   from typing import Any
   from lightning.pytorch import LightningDataModule
   import torch


   class MyDataModule(LightningDataModule):
       def __init__(self, parama=None, paramb="default"):
           self.parama = parama
           self.paramb = paramb
           super().__init__()
           self._metadata = None

       def _prepare_metadata(self):
           # todo: return model-init info, e.g. {"static_categorical": [...]}
           ...

       @property
       def metadata(self):
           if self._metadata is None:
               self._metadata = self._prepare_metadata()
           return self._metadata

       def _preprocess_data(self, series_idx: torch.Tensor) -> list[dict[str, Any]]:
           ...

       def setup(self, stage: str) -> None:
           ...

**Add a new metric / loss** (``extension_templates/metrics.py``) — subclass the
base for your metric type (``MultiHorizonMetric`` for point/generic;
``DistributionLoss`` / ``MultivariateDistributionLoss`` for distributional) and
implement ``loss``:

.. code-block:: python

   from pytorch_forecasting.metrics import MultiHorizonMetric


   class MyMetric(MultiHorizonMetric):
       def loss(self, y_pred, target):
           # todo: compute and return the unreduced loss tensor
           ...

**Add a new v1 network** (``extension_templates/v1/network/``) — v1 is still
maintained. Subclass the ``BaseModel`` variant that matches your model:
``BaseModel`` (no covariates, not autoregressive), ``BaseModelWithCovariates``
(static / time-varying covariates), ``AutoRegressiveBaseModel`` (autoregressive,
no covariates), or ``AutoRegressiveBaseModelWithCovariates`` (autoregressive with
covariates, e.g. DeepAR, TFT). Mandatory methods: ``__init__``, ``_pkg``,
``from_dataset`` (kept in v1 — reads sizes from a ``TimeSeriesDataSet``), and
``forward`` (returns via ``self.to_network_output(prediction=...)``):

.. code-block:: python

   class ExampleNetwork(BaseModel):
       def __init__(self, hidden_size: int = 16, **kwargs):
           self.save_hyperparameters()
           super().__init__(**kwargs)

       @classmethod
       def _pkg(cls):
           from extension_templates.v1.network._model_pkg import ExampleNetwork_pkg
           return ExampleNetwork_pkg

       @classmethod
       def from_dataset(cls, dataset, **kwargs):
           # derive dataset-dependent kwargs, then delegate
           return super().from_dataset(dataset, **kwargs)

       def forward(self, x: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
           # return via self.to_network_output(prediction=...)
           ...

Place reusable submodules in a ``layers/`` subfolder to keep the model file
focused.

Migrating a v1 model to v2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is migrating a model's *implementation* to the v2 architecture — a
contributor task. To migrate your own forecasting *code* from the v1 to the v2
API instead, see the migration guide in the :doc:`v2 documentation <api_v2>`.

Because the ``forward`` contract is identical, migrating is mostly a
re-organisation, not a rewrite:

1. **Move the network body** into a v2 ``model.py`` class inheriting the v2
   ``BaseModel``. The ``forward`` implementation usually transfers directly.
2. **Replace** ``from_dataset`` **size inference with** ``metadata``:

   .. code-block:: python

      # v1 — sizes come from the dataset via the factory classmethod
      @classmethod
      def from_dataset(cls, dataset, **kwargs):
          return super().from_dataset(dataset, **kwargs)

      # v2 — no from_dataset; sizes come from the DataModule's metadata dict
      def __init__(self, metadata, hidden_size=16, **kwargs):
          self.save_hyperparameters()
          super().__init__(**kwargs)
          # use metadata for embedding sizes, variable counts, etc.

3. **Extract tags and test parameters** into a ``model_pkg.py`` package class
   inheriting ``Base_pkg``, and point ``get_datamodule_cls()`` at the DataModule
   the model needs.
4. **Run** ``check_estimator`` **and register** the package class;
   ``TestAllPtForecastersV2`` then covers it automatically.

Large workstreams
~~~~~~~~~~~~~~~~~

Beyond the template kinds above, larger directions are in progress —
foundation-model integration, new train/test split strategies, ``torch.nn`` and
``MultiLoss`` loss adapters, DSIPTS / ``tslib`` integration, and a
pre-train / fine-tune API. These are **not yet built and have no per-kind
template**; until a direction lands (typically when a new template is added),
such work still follows the conventions above but needs design alignment with
the maintainers first. Open a discussion on the relevant issue or on Discord and
review the current `Roadmap`_ before starting.

Verifying and submitting
------------------------

Before opening a pull request, run the local checks described in `Setup / CI`_ —
the test suite (``pytest``), ``pre-commit run --all-files``, and a docs build
(``cd docs && make html``) — plus ``check_estimator`` on your package class (see
`Conventions every contribution follows`_). ReadTheDocs builds a preview for
every PR, linked in the PR checks.

Then submit: branch off ``main`` in your fork, open a pull request against
``main`` of ``sktime/pytorch-forecasting`` referencing the issue
(``Closes #NNNN``), and ensure all CI jobs (see `Setup / CI`_) pass before merge.
See :doc:`installation` (*Submitting pull request best practices*) for the PR
checklist, and the
`sktime developer guide <https://www.sktime.net/en/stable/developer_guide.html>`_
for the review process, enhancement proposals, and release procedure.

Roadmap
-------

Where development is heading is tracked in:

* `Roadmap 2026 (#1993) <https://github.com/sktime/pytorch-forecasting/issues/1993>`_
* `v2 work items (#1974) <https://github.com/sktime/pytorch-forecasting/issues/1974>`_
* `v2 documentation umbrella (#2304) <https://github.com/sktime/pytorch-forecasting/issues/2304>`_
* `sktime 2026 projects: pytorch-forecasting and dsipts <https://github.com/sktime/mentoring/blob/main/internships/projects_2026.md#pytorch-forecasting-and-dsipts>`_

Getting help
------------

* **Questions / bugs:** the
  `issue tracker <https://github.com/sktime/pytorch-forecasting/issues>`_.
* **Chat:** the community on
  `Discord <https://discord.com/invite/54ACzaFsn7>`_.
* **News:** sktime on
  `LinkedIn <https://www.linkedin.com/company/scikit-time/>`_.
