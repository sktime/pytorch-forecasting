Developer guide
===============

How to set up a development environment for ``pytorch-forecasting``, what the
code-quality checks actually run, and where to start when adding a model.

For the wider contribution process -- issue triage, the pull request workflow,
governance and the code of conduct -- see the
`sktime contributing guide <https://www.sktime.net/en/latest/get_involved/contributing.html>`_,
which ``pytorch-forecasting`` follows. This page covers only what is specific
to this package.

.. contents:: Contents
   :local:
   :depth: 2


Setting up a development environment
------------------------------------

``pytorch-forecasting`` supports Python 3.10 to 3.14.

Fork the repository on GitHub, then clone your fork and install it in editable
mode with the developer dependencies:

.. code-block:: console

   git clone https://github.com/<your-username>/pytorch-forecasting.git
   cd pytorch-forecasting

   python -m venv .venv
   source .venv/bin/activate     # Windows: .venv\Scripts\activate

   pip install -e ".[dev]"

Editable mode (``-e``) means your changes take effect without reinstalling.

The dependency sets
~~~~~~~~~~~~~~~~~~~

``pyproject.toml`` defines several extras. Which you need depends on what you
are working on:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - What it is for
   * - ``dev``
     - Contributor toolchain: ``pytest`` and plugins, ``pre-commit``, ``ruff``,
       ``mypy``, ``pylint``, and the Sphinx documentation stack. Start here.
   * - ``all_extras``
     - Every soft dependency. Needed to run the parts of the test suite that
       exercise optional functionality.
   * - ``tuning``
     - ``optuna`` and ``statsmodels``, for hyperparameter tuning.
   * - ``mqf2``
     - ``cpflows``, for the multivariate quantile loss.
   * - ``graph``
     - ``networkx``. Not currently used; kept for future work.

Soft dependencies are not required for core functionality, so most tests are
written to skip rather than fail when one is missing. If you are touching code
guarded by a soft dependency, install ``all_extras`` as well -- otherwise the
tests that cover your change will silently skip:

.. code-block:: console

   pip install -e ".[dev,all_extras]"

Install the pre-commit hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   pre-commit install

This runs the same checks CI runs, on the files you touch, before each commit.
Doing it now is cheaper than a round-trip through a red pipeline.


Running the tests
-----------------

.. code-block:: console

   python -m pytest                       # everything
   python -m pytest tests/test_data       # one directory
   python -m pytest -k encoder_normalizer # by name
   python -m pytest -n auto               # in parallel (pytest-xdist)

The full suite trains small models and takes a while. While iterating, run the
directory that matches what you changed and leave the rest to CI.

CI runs the suite on Ubuntu, macOS and Windows across Python 3.10 through
3.14, in two configurations: with all soft dependencies, and with none of them
(the ``no-softdeps`` job). A change that imports a soft dependency at module
level rather than inside a function will pass locally and fail there.


Code quality
------------

Formatting and linting are handled by `ruff <https://docs.astral.sh/ruff/>`_,
configured in ``pyproject.toml``:

.. code-block:: console

   ruff format .        # format
   ruff check . --fix   # lint, fixing what it can

The settings worth knowing before you argue with the tool:

* **Line length is 88.**
* **Target version is ``py310``**, so ``ruff`` will not suggest syntax newer
  than the oldest Python the package supports.
* **Import sorting is on** (``I``), with ``pytorch_forecasting`` as
  first-party, ``combine-as-imports``, and ``force-sort-within-sections``.
  Renaming a module can therefore change import order in files you did not
  otherwise touch; that hunk in your diff is expected.
* **Rule sets:** ``E``, ``F``, ``W``, ``C4``, ``S`` plus ``I`` and ``UP``.
  A handful of rules are disabled in ``extend-ignore`` -- ``E203``, ``E402``,
  ``E731``, ``E741`` and several ``C4`` rules -- each for a stated reason.

Notebooks are checked too, through
`nbQA <https://nbqa.readthedocs.io/>`_, so a tutorial notebook is held to the
same lint rules as a module.

The full hook list is in ``.pre-commit-config.yaml``: trailing whitespace,
end-of-file fixing, YAML validity, an AST parse check, ``ruff``,
``ruff-format``, and the nbQA equivalents.

Docstrings
~~~~~~~~~~

Public functions, classes and methods carry numpydoc-style docstrings with
``Parameters`` and ``Returns`` sections. They are rendered into the API
reference, so a missing or malformed section shows up in the built docs.


Building the documentation
--------------------------

.. code-block:: console

   cd docs
   make html

The result is in ``docs/build/html``. Open ``index.html`` in a browser.

If you add a page, add it to the ``toctree`` in ``docs/source/index.rst`` --
otherwise Sphinx builds it but nothing links to it.


Adding a model or a data module
-------------------------------

Do not start from a blank file. The ``extension_templates`` directory holds
fill-in templates with ``todo`` comments marking what you must supply. Copy one
to a suitable location, rename it, and fill it in.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Template
     - Use for
   * - ``extension_templates/v1/network/``
     - A v1 model: the network plus its package class.
   * - ``extension_templates/v2/model_simple/``
     - A v2 model: ``model.py`` and ``model_pkg.py``.
   * - ``extension_templates/v2/data_module/``
     - A v2 data module and its private dataset.
   * - ``extension_templates/metrics.py``
     - A custom metric.

``extension_templates/v2/README.md`` explains the v2 architecture, and is
worth reading before filling anything in.

The model/package split
~~~~~~~~~~~~~~~~~~~~~~~

Every estimator is two classes:

* the **model** -- the network, forward pass and training logic;
* the **package** -- metadata, tags, capabilities and test parameters.

They are linked in both directions: ``MyModel._pkg()`` returns the package
class, and ``MyModel_pkg.get_cls()`` returns the model class. The split exists
so that the framework can read an estimator's metadata without importing its
dependencies, which is what keeps import time and memory down.

Package files are named ``_<model>_pkg.py`` for v1 and ``_<model>_pkg_v2.py``
for v2, and the package class is named after the model with a ``_pkg`` or
``_pkg_v2`` suffix. ``test_all_estimators_v2`` derives the expected package
name from the class name, so a mismatch is caught by the test suite.

Test parameters
~~~~~~~~~~~~~~~

The package class supplies ``get_test_params``. Everything a new estimator
needs to be swept up by the shared estimator tests comes from there -- you do
not write per-model test files for the standard checks. Make the parameters
small: they are instantiated and trained on every CI run, on every supported
Python version and operating system.


Two API versions
----------------

The package currently carries both a stable v1 API and a v2 API under
development. v2 is not production-ready, and the modules that belong to it warn
on import to say so.

The two are documented separately -- :doc:`api` and :doc:`api_v2`,
:doc:`models` and :doc:`models_v2` -- and the ``_v2`` suffix on a module or a
class marks which it belongs to. When fixing a bug, check whether the same bug
exists in the other version; they share concepts but not code.


Changelog
---------

Release notes are generated from merged pull requests by
``build_tools/changelog.py``, so your PR title becomes the changelog entry.
Write it as the line you would want a user to read.
