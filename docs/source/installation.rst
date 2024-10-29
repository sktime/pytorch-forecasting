Installation
============

``pytorch-forecasting`` currently supports:

* Python versions 3.8, 3.9, 3.10, 3.11, and 3.12.
* Operating systems : ... ... and ...

Installing pytorch-forecasting
------------------------------

``pytorch-forecasting`` is a library build off of the popular deep learning framework ``pytorch`` and
heavily uses the Pytorch Lightning library ``lightning`` for ease of training and multiple GPU usage.

You'll need to install ``pytorch`` along or before with ``pytorch-forecasting`` in order to get a working
install of this library.

If you are working Windows, you can install PyTorch with

.. code-block:: bash

    pip install torch -f https://download.pytorch.org/whl/torch_stable.html

.. note::
  It is recommended to visit the Pytorch official page https://pytorch.org/get-started/locally/#start-locally to
  figure out which version of ``pytorch`` best suits your machine if you are
  unfamiliar with the library.

Otherwise, you can proceed with:

.. code-block:: bash
    pip install pytorch-forecasting


Alternatively, to install the package via ``conda``:
.. code-block:: bash
    conda install pytorch-forecasting pytorch>=1.7 -c pytorch -c conda-forge

PyTorch Forecasting is now installed from the conda-forge channel while PyTorch is install from the pytorch channel.

To install ``pytorch-forecasting`` with the use of the MQF2 loss (multivariate quantile loss), run:

.. code-block:: bash
    pip install pytorch-forecasting[mqf2]


To install the Pytorch Lightning library, please visit their `official page <https://lightning.ai/docs/pytorch/stable/starter/installation.html>`__ or run:

.. code-block:: bash
    pip install lightning


Obtaining a latest ``pytorch-forecasting`` version
--------------------------------------------------

This type of installation obtains a latest static snapshot of the repository, with
various features that are not published in a release. It is mainly intended for developers
that wish to build or test code using a version of the repository that contains
all of the latest or current updates.

.. code-block:: bash

    pip install git+https://github.com/sktime/pytorch-forecasting.git


To install from a specific branch, use the following command:

.. code-block:: bash

    pip install git+https://github.com/sktime/pytorch-forecasting.git@<branch_name>


Contributing to ``pytorch-forecasting``
---------------------------------------

Contributions to PyTorch Forecasting are very welcome! You do not have to be an expert in deep learning
to contribute. If you find a bug - fix it! If you miss a feature - propose it!

To obtain an editible version ``pytorch-forecasting`` for development or contributions,
you will need to set up:

* a local clone of the ``pytorch-forecasting`` repository.
* a virtual environment with an editable install of ``pytorch-forecasting`` and the developer dependencies.

The following steps guide you through the process:

Creating a fork and cloning the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.  Fork the `project
    repository <https://github.com/sktime/pytorch-forecasting>`__ by
    clicking on the 'Fork' button near the top right of the page. This
    creates a copy of the code under your GitHub user account. For more
    details on how to fork a repository see `this
    guide <https://help.github.com/articles/fork-a-repo/>`__.

2.  `Clone <https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository>`__
    your fork of the pytorch-forecasting repo from your GitHub account to your local
    disk:

    .. code:: bash

      git clone git@github.com:<username>/sktime/pytorch-forecasting.git
      cd pytorch-forecasting

    where :code:`<username>` is your GitHub username.

3.  Configure and link the remote for your fork to the upstream
    repository:

    .. code:: bash

      git remote -v
      git remote add upstream https://github.com/sktime/pytorch-forecasting.git

4.  Verify the new upstream repository you've specified for your fork:

    .. code:: bash

      git remote -v
      > origin    https://github.com/<username>/sktime/pytorch-forecasting.git (fetch)
      > origin    https://github.com/<username>/sktime/pytorch-forecasting.git (push)
      > upstream  https://github.com/sktime/pytorch-forecasting.git (fetch)
      > upstream  https://github.com/sktime/pytorch-forecasting.git (push)

Setting up an editible virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Set up a new virtual environment. Our instructions will go through the commands to set up a ``conda`` environment which is recommended for ``pytorch-forecasting`` development.
The process will be similar for ``venv`` or other virtual environment managers.

  .. warning::
       Using ``conda`` via one of the commercial distributions such as Anaconda
       is in general not free for commercial use and may incur significant costs or liabilities.
       Consider using free distributions and channels for package management,
       and be aware of applicable terms and conditions.

In the ``conda`` terminal:

2. Navigate to your local pytorch-forecasting folder, :code:`cd pytorch-forecasting` or similar

3. Create a new environment with a supported python version: :code:`conda create -n pytorch-forecasting-dev python=3.11` (or :code:`python=3.12` etc)

   .. warning::
       If you already have an environment called ``pytorch-forecasting-dev`` from a previous attempt you will first need to remove this.

4. Activate the environment: :code:`conda activate pytorch-forecasting-dev`

5. Build an editable version of pytorch-forecasting.
In order to install only the dev dependencies, :code:`pip install -e ".[dev]"`
If you also want to install soft dependencies, install them individually, after the above,
or instead use: :code:`pip install -e ".[all_extras,dev]"` to install all of them.

Contribution Guidelines and Recommendations
-------------------------------------------

Submitting pull request best practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure that maintainers and other developers are able to help your issues or
review your contributions/pull requests, please read the following guidelines below.

* Open issues to discuss your proposed changes before starting pull requests.
  This ensures that other developers or maintainers have adequete context/knowledge
  about your future contribution so that it can be swiftly integrated into the code base.

* Adding context tags to the PR title.
  This will greatly help categorize different types of pull requests without having
  to look at the full title. Usually tags that start with either [ENH] - Enhancement:
  adding a feature, or improving code, [BUG] - Bugfixes, [MNT] - CI: test framework, [DOC] -
  Documentation: writing or improving documentation or docstrings.

* Adding references to other links or pull requests
  This helps to add context about previous or current issues/prs that relate to
  your contribution. This is done usually by including a full link or a hash tag '#1234'.

Technical Design Principles
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When writing code for your new feature, it is recommended to follow these
technical design principles to ensure compatability between the feature and the library.

* Backward compatible API if possible to prevent breaking code.
* Powerful abstractions to enable quick experimentation. At the same time, the abstractions should
  allow the user to still take full control.
* Intuitive default values that do not need changing in most cases.
* Focus on forecasting time-related data - specificially timeseries regression and classificiation.
  Contributions not directly related to this topic might not be merged. We want to keep the library as
  crisp as possible.
* Install ``pre-commit`` and have it run on every commit that you make on your feature branches.
  This library requires strict coding and development best practices to ensure the highest code quality.
  Contributions or pull requests that do not adhere to these standards will not likely be merged until fixed.
  For more information on ``pre-commit`` you can visit `this page <https://www.sktime.net/en/stable/developer_guide/coding_standards.html#using-pre-commit>`__
* Always add tests and documentation to new features.
