API v2
======

.. warning::
    Please note that the v2 modules are currently in active-development and is in beta right now, so please use this API with caution.
    See v1 documentation :doc:`here <api>` - it is stable and can be used in the production pipelines.

.. currentmodule:: pytorch_forecasting

We are currently developing version 2 of PyTorch Forecasting. The primary objective of this redesign is to improve the software architecture and provide a more intuitive workflow for developers and data scientists.

The key structural changes and design philosophies driving V2 include:

* **Decoupling Models and Data Structures:** In the original API, models are heavily tightly coupled with the :py:class:`~data.timeseries.TimeSeriesDataSet` class. V2 systematically reduces this strict dependency. By decoupling the forecasting models from specific data-handling classes, the models become more modular and can interface more seamlessly with standard PyTorch tensors, data loaders, and external data pipelines.
* **Unified API with Exchangeable Models:** V2 introduces a high-level Package wrapper providing a standardized ``fit()`` and ``predict()`` workflow, making it effortless to swap between different forecasting architectures. Crucially, this does not replace the PyTorch Lightning interface—advanced users retain full access to interact with the underlying models directly at a lower level.
* **Simplified User Journey:** Consequently, these architectural changes drastically reduce the amount of boilerplate code required to set up data, initialize models, and generate predictions, allowing users to move from raw data to forecasting more efficiently.

The New Layered Architecture
----------------------------

To achieve this decoupling and streamline the user journey, API-v2 introduces a strict, four-layered structure for the entire model training and prediction workflow:

* **D1 Layer (Dataset Layer):** A foundational dataset layer responsible for ingesting raw data and converting it into PyTorch tensors. It also extracts and stores fundamental metadata, such as static variables.
* **D2 Layer (DataModule Layer):** Implemented as a PyTorch Lightning ``LightningDataModule``, this layer handles all data preprocessing, instantiates the dataloaders, and collects the necessary structural information (e.g., the number of categorical variables) required to properly initialize the forecasting models.
* **Model Layer:** Implemented as a PyTorch Lightning ``LightningModule``, this layer contains the pure PyTorch implementation of the core forecasting algorithms. It remains entirely agnostic to the complexities of the data ingestion pipelines.
* **Package Layer:** Acting as a higher-level wrapper around the underlying layers, this component manages the orchestration of the workflow. It handles the simultaneous initialization of the layers, exposes the unified ``fit`` and ``predict`` interfaces, and houses the fixtures utilized for testing.

Metrics in V2
-------------

Currently, API-v2 leverages the exact same metrics suite established in API-v1 to ensure predictive consistency and reliability during the transition. However, to further align with the broader PyTorch ecosystem, we plan to introduce native support for standard ``torch.nn`` metrics and loss functions in future releases.


We Need Your Feedback
---------------------

The API-v2 is being built for the community, and your input is critical to ensuring it meets your needs. We encourage you to try out the new modules, test them, and let us know what changes, features, or refinements you would like to see before the final release.

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
