.. _developer_guide:

Developer Guide
================

.. currentmodule:: pytorch_forecasting

This guide is for contributors and developers who want to work on
``pytorch-forecasting`` itself. It covers environment setup, code quality
tooling, the testing pipeline, and pointers for extending the library with
new models or data modules.

If you are looking for user-facing documentation, see :doc:`getting-started`
and the :ref:`Tutorials <tutorials>`.


Setting up the development environment
---------------------------------------

**1. Fork and clone the repository**

Start by forking `sktime/pytorch-forecasting
<https://github.com/sktime/pytorch-forecasting>`_ on GitHub, then clone your
fork locally::

    git clone git@github.com:<your-username>/pytorch-forecasting.git
    cd pytorch-forecasting

**2. Create a virtual environment**

We recommend using ``venv`` or ``conda`` to isolate your dependencies.

With ``venv``::

    python -m venv .venv
    source .venv/bin/activate   # Linux / macOS
    .venv\Scripts\activate      # Windows

With ``conda``::

    conda create -n ptf-dev python=3.11
    conda activate ptf-dev

.. note::

   The package supports Python 3.10 through 3.14. Python 3.11 is a safe
   default for development.

**3. Install the package in editable mode with dev extras**

This pulls in all development and testing dependencies (``pytest``,
``pre-commit``, ``ruff``, ``sphinx``, etc.)::

    pip install -e ".[dev,all_extras]"

On **macOS**, you may also need to install ``libomp`` for PyTorch::

    brew install libomp

**4. Install pre-commit hooks**

pre-commit runs the linters and formatters automatically on every commit::

    pre-commit install

After this, every ``git commit`` will run the checks on the files you changed.
You can also run them manually at any time::

    pre-commit run --all-files


**5. Verify the setup**

Run the test suite to confirm everything is working::

    python -m pytest

A passing run means your environment is ready.


Code quality
-------------

We enforce a consistent code style through automated tooling. The CI pipeline
runs these checks on every pull request, and your PR will not be merged until
they pass.

pre-commit hooks
~~~~~~~~~~~~~~~~~

The ``.pre-commit-config.yaml`` at the root of the repository defines the
hooks that run on each commit:

* **trailing-whitespace** — removes invisible trailing spaces.
* **end-of-file-fixer** — ensures every file ends with exactly one newline.
* **check-yaml** — validates YAML syntax.
* **check-ast** — validates Python syntax (catches broken ``import`` statements
  before they reach CI).
* **ruff** — a fast Python linter that replaces ``flake8``, ``isort``, and
  ``pyupgrade`` in a single tool. Configured to auto-fix issues where possible.
* **ruff-format** — auto-formats code, replacing ``black`` with a
  compatible formatter.
* **nbqa-black** / **nbqa-ruff** — applies the same checks to Jupyter
  notebooks.

To run them manually against all files::

    pre-commit run --all-files

Or against specific files::

    pre-commit run --files path/to/file.py

ruff configuration
~~~~~~~~~~~~~~~~~~~

Ruff is configured in ``pyproject.toml`` under ``[tool.ruff]``:

* **Line length**: 88 characters.
* **Target version**: Python 3.10.
* **Selected rules**: ``E`` (pycodestyle), ``F`` (pyflakes), ``W`` (warnings),
  ``I`` (isort import sorting), ``UP`` (pyupgrade), ``C4`` (comprehensions),
  ``S`` (bandit security checks).
* **Quote style**: double quotes.
* **Import sorting**: first-party imports (``pytorch_forecasting``) are grouped
  separately.

When ruff auto-fixes an issue (e.g., reordering imports), it modifies the file
in place. Review the changes with ``git diff`` before committing.

Docstring conventions
~~~~~~~~~~~~~~~~~~~~~~

We follow the `NumPy docstring format
<https://numpydoc.readthedocs.io/en/latest/format.html>`_. Every public class
and function should have a docstring with at minimum:

* A one-line summary.
* A ``Parameters`` section listing all arguments with types and descriptions.
* A ``Returns`` section for functions that return values.

Example::

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the forward pass of the model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary of input tensors produced by the data module.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the model predictions.
        """


Testing
--------

The test suite uses ``pytest`` and lives in two directories:

* ``tests/`` — integration and end-to-end tests.
* ``pytorch_forecasting/tests/`` — unit tests collocated with the source.

Configuration is in ``pyproject.toml`` under ``[tool.pytest.ini_options]``.

**Running all tests**::

    python -m pytest

**Running a specific test file**::

    python -m pytest tests/test_models/test_base_model_v2.py

**Running tests in parallel** (requires ``pytest-xdist``)::

    python -m pytest -n auto

**Coverage report** is generated automatically and printed to the terminal.
An HTML report is written to ``htmlcov/``.

Continuous integration
~~~~~~~~~~~~~~~~~~~~~~~

The CI pipeline (defined in ``.github/workflows/test.yml``) runs the following
jobs on every pull request:

1. **code-quality** — runs ``pre-commit`` on changed files.
2. **pytest-nosoftdeps** — runs the test suite *without* optional dependencies
   across Python 3.10–3.14 on Linux, macOS, and Windows.
3. **pytest** — runs the full test suite *with* all extras installed, across
   the same matrix.
4. **run-notebook-tutorials** — executes the tutorial notebooks to catch
   regressions.
5. **test-deps-2025** — tests against a pinned dependency snapshot to catch
   breakages from upstream releases.

All jobs must pass before a PR can be merged.


Building the documentation
---------------------------

The documentation is built with `Sphinx <https://www.sphinx-doc.org/>`_ and
hosted on `Read the Docs <https://pytorch-forecasting.readthedocs.io/>`_.

Source files live in ``docs/source/`` and are written in reStructuredText
(``.rst``).

To build locally::

    cd docs
    make html

The output is written to ``docs/build/html/``. Open ``index.html`` in a
browser to preview.

You need the ``docs`` extras installed::

    pip install -e ".[docs]"


Git workflow
-------------

We follow a standard fork-and-branch workflow:

1. **Fork** the repository on GitHub.
2. **Create a feature branch** from ``main``::

       git checkout -b feat/my-feature

3. **Make your changes**, committing with clear messages. The pre-commit hooks
   will enforce formatting on each commit.
4. **Push** to your fork and open a pull request against ``sktime:main``.
5. **Address review feedback** by pushing additional commits to your branch.

**Branch naming conventions**:

* ``feat/`` — new features or enhancements.
* ``fix/`` — bug fixes.
* ``doc/`` — documentation changes.
* ``mnt/`` — maintenance (CI, dependencies, refactoring).

**Commit messages** should be concise and start with a tag:
``[ENH]``, ``[BUG]``, ``[DOC]``, ``[MNT]``, ``[TST]``.

Example: ``[ENH] Add NLinear model to v2 package layer``


Extending the library
----------------------

Adding a new model (v2)
~~~~~~~~~~~~~~~~~~~~~~~~

The v2 architecture separates the neural network from the framework metadata.
Adding a new model requires two files:

* ``model.py`` — the ``LightningModule`` with the network and training logic.
* ``model_pkg.py`` — the ``Base_pkg`` subclass with metadata, tags, and
  factory methods.

Ready-to-copy templates are in the
`extension_templates/v2/model_simple/
<https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v2/model_simple>`_
directory.

After implementing your model, validate it with the built-in checks::

    from pytorch_forecasting.utils._estimator_checks import check_estimator
    from pytorch_forecasting.models.my_model import MyModel

    check_estimator(MyModel)

See the `extension templates README
<https://github.com/sktime/pytorch-forecasting/blob/main/extension_templates/v2/README.md>`_
for the full walkthrough.

Adding a new data module (v2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only create a new data module if the existing ones
(:py:class:`~data.data_module.EncoderDecoderTimeSeriesDataModule`,
:py:class:`~data.data_module.TslibDataModule`) cannot produce the tensor
format your model expects.

Templates are in
`extension_templates/v2/data_module/
<https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v2/data_module>`_.

Adding a new model (v1)
~~~~~~~~~~~~~~~~~~~~~~~~

For the legacy v1 API, model templates are in
`extension_templates/v1/network/
<https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v1/network>`_.

Adding new metrics
~~~~~~~~~~~~~~~~~~~

Metric templates are in
`extension_templates/metrics.py
<https://github.com/sktime/pytorch-forecasting/blob/main/extension_templates/metrics.py>`_.
