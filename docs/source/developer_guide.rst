.. _developer_guide:

Developer Guide
===============

This guide is for people who want to **develop** ``pytorch-forecasting``,
meaning contributing code rather than just using the library. It is written for
the **v2** API: v2 is where all new development happens, and new estimators
should be contributed there. **v1 is in maintenance mode** and receives bug
fixes only; see :doc:`api` for the v1 reference and :doc:`migration_v1_to_v2`
for moving off it.

``pytorch-forecasting`` is part of the `sktime <https://www.sktime.net>`_
ecosystem. Project-wide conventions that are not specific to this package
(coding standards, the git and reviewer workflow, enhancement proposals,
deprecation policy, and governance) follow the sktime developer documentation
and are linked rather than repeated here:

* sktime developer guide: https://www.sktime.net/en/stable/developer_guide.html
* sktime contributing guide: https://www.sktime.net/en/latest/get_involved/contributing.html

.. note::

   Using the library to build forecasts, rather than changing its source? See
   the :doc:`installation guide <installation>`. If you are migrating your own
   v1 forecasting *code* to the v2 API, see the :doc:`migration_v1_to_v2` guide.
   This developer guide is about changing the library itself.

Overview
--------

A contribution to ``pytorch-forecasting`` is usually a new **estimator**: a v2
model, a v2 data module, or a metric. Whatever you add follows a small set of
shared conventions and, where one exists, an official extension template. This
guide walks through setting up your environment, the architecture (so the
conventions make sense), and how to add, verify, and submit a contribution.

Setup / CI
----------

Setup
~~~~~

Follow :doc:`installation` (see *Contributing to pytorch-forecasting*) to create
a fork, clone it, and set up an editable virtual environment with the developer
dependencies: ``pip install -e ".[dev]"`` (add ``",all_extras"`` for the soft
dependencies used by some estimators). Then install the pre-commit hooks:

.. code-block:: bash

   pre-commit install

The ``[dev]`` extra installs the linters, ``pytest`` (+ ``pytest-xdist`` /
``pytest-cov`` / ``pytest-sugar``), and the docs build stack.

Code quality
~~~~~~~~~~~~

Pre-commit is the single entry point for code style. After ``pre-commit install``
the following hooks run automatically on every ``git commit``:

* **trailing-whitespace**: strips trailing whitespace.
* **end-of-file-fixer**: ensures files end with a newline.
* **check-yaml**: validates YAML syntax.
* **check-ast**: validates Python syntax.
* **ruff** (``--fix``): linting with auto-fix; runs the rule sets described
  below.
* **ruff-format**: opinionated formatter (line length 88, double quotes, space
  indent).
* **nbqa-black**: applies Black formatting to Jupyter notebooks.
* **nbqa-ruff**: applies ruff linting to Jupyter notebooks.
* **nbqa-check-ast**: validates notebook cell syntax.

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
docstrings. For the full coding standards (naming conventions, type
annotations, and import organisation), follow the
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

**The v2 test suite.** ``TestAllPtForecastersV2`` discovers estimators through
the ``all_objects`` registry, so a correctly registered model is picked up
automatically and needs no per-model test file. It checks the interface
contract: the docstring doctests, training and prediction end to end, a
checkpoint round-trip, the output shapes of each predict mode, and the package
naming convention. The parameter sets it runs come from
``get_test_train_params`` on the package class.

**Running the checks locally.** ``check_estimator`` runs that same suite against
a single estimator, and works before it is registered.

.. code-block:: python

   from pytorch_forecasting.utils import check_estimator

   results = check_estimator(MyModel_pkg)                # summary printout
   check_estimator(MyModel_pkg, raise_exceptions=True)   # raise, for debugging

Pass either the model class or the package class; the model class resolves to
its package automatically. The result maps each ``test[fixture]`` to
``"PASSED"`` or to the exception raised. Failures are collected rather than
raised unless ``raise_exceptions=True``.

**Common pitfalls.**

* **A wrong** ``object_type`` **tag passes silently.** ``"forecaster_v2"`` instead
  of ``"forecaster_pytorch_v2"`` matches no registered scitype, so no test class
  is found and nothing runs, yet the call still prints ``All tests PASSED!``.
  Check ``len(results) > 0`` before trusting a green run.
* **The** ``capability:*`` **tags are never verified.** Declaring
  ``"capability:multivariate": True`` does not cause multivariate behaviour to be
  tested; no test reads these tags.
* **Numerical correctness is not checked.** The suite asserts on shapes, not on
  values, so a model returning plausible nonsense passes. Add your own test, for
  example that a constant input series produces a constant forecast.
* **Only** ``SMAPE`` **is used as the loss**, and declaring ``info:pred_type`` to
  change that currently breaks fixture generation. See the note below.

.. note::

   **Known gap, remove this note once resolved.** The loss-compatibility matrix
   in ``pytorch_forecasting/tests/_loss_mapping.py`` selects test losses from the
   ``info:pred_type`` and ``info:y_type`` tags, but the surrounding wiring is
   v1-only. No v2 package declares ``info:pred_type``, so no loss is selected and
   every v2 model is tested with the ``SMAPE`` fallback. Declaring the tag does
   not help: the selection path then calls ``get_base_test_params``, which
   ``Base_pkg`` does not define, so fixture generation raises ``AttributeError``.
   Note that the v2 model template still lists ``info:pred_type``. Until this is
   wired up for v2, add the loss to a ``get_test_train_params`` entry instead,
   for example ``{"loss": QuantileLoss()}``.

Docs
~~~~

The documentation is built with Sphinx:

.. code-block:: bash

   cd docs && make html

The rendered HTML lands in ``docs/build/html``. Build it locally before opening
a pull request that touches documentation, and check that your change adds no
new warnings.

Continuous integration
~~~~~~~~~~~~~~~~~~~~~~

On every push or pull request targeting ``main``,
``.github/workflows/test.yml`` runs pre-commit on the changed files, the full
``pytest`` suite both with and without the soft dependencies, a run pinned to a
recent dependency snapshot, and the example notebooks. All of these must pass
before a pull request can be merged.

Extension library
~~~~~~~~~~~~~~~~~

The `extension_templates/ <https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates>`_
directory holds the official copy-and-fill templates for every recognised
contribution type (v2 model, v2 data module, metric). See
`Adding a contribution`_ for details and code skeletons.

Architecture
------------

v2 splits the v1 monolith, in which ``TimeSeriesDataSet`` handled ingestion,
preprocessing, and batching and the model was built from it via
``Model.from_dataset(...)``, into four decoupled layers (Dataset, DataModule,
Model, Package), where the model is driven by a ``metadata`` dict instead of
``from_dataset`` (full reference: :doc:`api_v2`).

.. mermaid::

   flowchart TD
       raw["Raw data (pandas DataFrame)"] -->|raw| d1["D1: TimeSeries"]
       d1 -->|"tensors + base metadata"| d2["D2: LightningDataModule"]
       d2 -->|"metadata + dataloaders"| model["Model: BaseModel<br/>(LightningModule)"]
       user(["User"]) ==>|"high-level API:<br/>fit / predict"| pkg["Package: Base_pkg"]
       pkg -.->|orchestrates| d1
       user -->|"low-level API:<br/>drive the 3 stages via Lightning"| d1

Runnable end-to-end examples are in the v2 tutorials
:doc:`tutorials/ptf_V2_example` and :doc:`tutorials/tslib_v2_example`.

.. warning::

   The v2 modules are under active development and are in beta. Use the v2 API
   with caution; it is not yet production-ready. The stable v1 API remains
   available for production pipelines. Feedback is welcome in
   `issue #1736 <https://github.com/sktime/pytorch-forecasting/issues/1736>`_.

The model-implementation differences from v1 (data layer split, ``from_dataset``
to ``metadata``, the new package base, unchanged ``forward``) are compared item
by item in :doc:`migration_v1_to_v2`.

Adding a contribution
---------------------

A contribution is usually a new estimator: a model, a data module, or a metric.
Every v2 estimator follows the same conventions (below); then, for each kind,
there is an official template to copy and fill in.

Conventions every contribution follows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Package class inheriting** ``Base_pkg`` **with a** ``_tags`` **dictionary**
  describing the estimator's name, compute cost, prediction type, target type,
  authors, capabilities, and soft dependencies.
* **Registered for auto-discovery** so the ``all_objects`` registry and the
  ``TestAllPtForecastersV2`` test suite find it automatically, so no
  per-estimator test file is needed. (v2 estimators carry the ``object_type``
  tag ``forecaster_pytorch_v2``, distinguishing them from v1's
  ``forecaster_pytorch_v1``.)
* **Passes** ``check_estimator`` (run it before opening a PR; see `Test`_).
* **Provides** ``get_test_train_params``: the first entry must be ``{}`` so
  default construction is tested, and every entry must be low-compute (minimal
  sequence length, tiny hidden sizes) so CI does not time out.
* **Guarded soft-dependency imports**: import any non-core package *inside the
  method that needs it*, not at module top level, so the core still imports.

The ``_tags`` dictionary is how the framework understands your estimator (keys
from the template at ``extension_templates/v2/model_simple/model_pkg.py``):

.. code-block:: python

   _tags = {
       # Model name: MUST match the model class name exactly.
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

Official templates
~~~~~~~~~~~~~~~~~~

For each kind there is an official template in ``extension_templates/``. Copy
it, rename it, and work through the ``todo`` comments. Templates are scaffolds,
not base classes to import.

**Add a new v2 model** (`extension_templates/v2/model_simple/ <https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v2/model_simple>`_):
two files. ``model.py`` holds the network (mandatory: ``_pkg`` and ``forward``;
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
``BaseModel``; it handles the metadata and initialisation boilerplate shared by
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

Place reusable submodules in a ``layers/`` subfolder to keep the model file
focused.

**Add a new v2 data module** (`extension_templates/v2/data_module/ <https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v2/data_module>`_):
a D2 ``LightningDataModule`` (mandatory: ``_prepare_metadata``, the ``metadata``
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

**Add a new metric / loss** (`extension_templates/metrics.py <https://github.com/sktime/pytorch-forecasting/blob/main/extension_templates/metrics.py>`_):
subclass the base for your metric type (``MultiHorizonMetric`` for point/generic;
``DistributionLoss`` / ``MultivariateDistributionLoss`` for distributional) and
implement ``loss``:

.. code-block:: python

   from pytorch_forecasting.metrics import MultiHorizonMetric


   class MyMetric(MultiHorizonMetric):
       def loss(self, y_pred, target):
           # todo: compute and return the unreduced loss tensor
           ...

.. note::

   New models should be contributed to v2. v1 is in maintenance mode: it accepts
   bug fixes, but new networks are no longer added there. If you have a v1 model
   you want in the library, port it to v2 following `Migrating a v1 model to
   v2`_.

Migrating a v1 model to v2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is migrating a model's *implementation* to the v2 architecture, a
contributor task. To migrate your own forecasting *code* from the v1 to the v2
API instead, see the :doc:`migration_v1_to_v2` guide.

Because the ``forward`` contract is identical, migrating is mostly a
re-organisation, not a rewrite:

1. **Move the network body** into a v2 ``model.py`` class inheriting the v2
   ``BaseModel``. The ``forward`` implementation usually transfers directly.
2. **Replace** ``from_dataset`` **size inference with** ``metadata``:

   .. code-block:: python

      # v1: sizes come from the dataset via the factory classmethod
      @classmethod
      def from_dataset(cls, dataset, **kwargs):
          return super().from_dataset(dataset, **kwargs)

      # v2: no from_dataset; sizes come from the DataModule's metadata dict
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

Beyond the template kinds above, larger directions are in progress:
foundation-model integration, new train/test split strategies, ``torch.nn`` and
``MultiLoss`` loss adapters, DSIPTS / ``tslib`` integration, and a
pre-train / fine-tune API. These are **not yet built and have no per-kind
template**; until a direction lands (typically when a new template is added),
such work still follows the conventions above but needs design alignment with
the maintainers first. Open a discussion on the relevant issue or on Discord and
review the current `Roadmap`_ before starting.

Submitting
----------

Run the local checks in `Setup / CI`_ before opening a pull request. Then branch
off ``main`` in your fork, open a pull request against ``main`` of
``sktime/pytorch-forecasting`` referencing the issue (``Closes #NNNN``), and
ensure all CI jobs pass before merge. ReadTheDocs builds a preview for every PR,
linked in the PR checks.

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
