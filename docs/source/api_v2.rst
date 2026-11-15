API v2
======

.. warning::
    Please note that the v2 modules are currently in active development and are in beta. Please use this API with caution.
    See the stable v1 documentation :doc:`here <api>` for production use.

.. currentmodule:: pytorch_forecasting

We are currently developing version 2 of PyTorch Forecasting. The primary objective of this redesign is to improve the software architecture and provide a more intuitive workflow for developers and data scientists.

The key structural changes and design philosophies driving V2 include:

* **Decoupling Models and Data Structures:** In the original API, models are tightly coupled with the :py:class:`~data.timeseries.TimeSeriesDataSet` class. V2 reduces this dependency, allowing forecasting models to work with a more flexible data interface and standard PyTorch components.
* **Unified Forecaster API:** V2 introduces a high-level Forecaster interface that provides standardized ``fit()`` and ``predict()`` workflows across forecasting architectures. Advanced users can still interact directly with the underlying PyTorch Lightning models.
* **Simplified User Journey:** These architectural changes reduce the boilerplate required to configure data, initialize models, train forecasting models, and generate predictions.

The New Layered Architecture
----------------------------

API v2 organizes the forecasting workflow into four main components:

* **TimeSeries (Data Container):** The :py:class:`~data.timeseries._timeseries_v2.TimeSeries` class is the standard input and output data structure in V2. It holds tabular time series data together with its schema and metadata. It is a data container rather than a conventional data ingestion or preprocessing layer.
* **DataModule Layer:** Implemented using PyTorch Lightning's ``LightningDataModule``, this layer handles data preprocessing, creates dataloaders, and collects the metadata required to initialize forecasting models. The ``with_data()`` method allows users to create a new DataModule instance with different data while retaining the original configuration.
* **Model Layer:** Implemented as a PyTorch Lightning ``LightningModule``, this layer contains the core forecasting algorithms and their training logic.
* **Forecaster Layer:** This high-level layer coordinates the DataModule and Model layers and exposes a unified ``fit()`` and ``predict()`` interface. It simplifies the workflow while allowing access to the underlying model when needed.

Predictions in V2
-----------------

In the standard prediction workflow, the Forecaster returns predictions as a :py:class:`~data.timeseries._timeseries_v2.TimeSeries` object. This provides a consistent data structure for both input time series and forecast output.

Predictions can be converted to a pandas DataFrame using the ``to_pandas()`` method:

.. code-block:: python

    predictions = forecaster.predict()
    predictions_df = predictions.to_pandas()

Metrics in V2
-------------

Currently, API v2 uses the same metrics suite as API v1 to maintain consistency during the transition. Native support for standard ``torch.nn`` metrics and loss functions may be introduced in future releases.

We Need Your Feedback
---------------------

API v2 is being developed for the community, and your feedback is important. We encourage you to try the new modules, test them, and share suggestions for improvements before the final release.

Join the discussion and track our progress on GitHub:

* *Feedback & Suggestions:* `API-v2 Development Issue <https://github.com/sktime/pytorch-forecasting/issues/1736>`_
* *Future Plans:* `PyTorch Forecasting Roadmap <https://github.com/sktime/pytorch-forecasting/issues/1993>`_

**Find the links to specific parts of the API below:**

.. toctree::
    :maxdepth: 2

    Data <data_v2>
    Models <models_v2>
    Package <pkg_v2>
    Metrics <metrics>
    Utils <utils>
    Tutorials <tutorials_v2>
