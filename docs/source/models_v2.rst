Models v2
==========

.. warning::
    Please note that the v2 modules are currently in active-development and is in beta right now, so please use this API with caution.
    See complete documentation for v2 API :doc:`here <api_v2>` and stable v1 documentation :doc:`here <api>`.

.. _models:

.. currentmodule:: pytorch_forecasting

The forecasting models in the V2 ecosystem are designed with a strict emphasis on modularity and separation of concerns. The architecture decouples algorithmic logic from data processing, ensuring that models act as pure, data-agnostic PyTorch Lightning instances.

Architecture
------------
The v2 models in ``pytorch-forecasting`` are separated into two distinct sub-layers:

* **The M Layer (Model):** The core ``torch`` neural network implementation, inheriting from PyTorch Lightning's ``LightningModule``. Designed for experienced developers, this layer allows you to bypass the package wrapper to build fully custom training, testing, and prediction pipelines.
    * **Learn more:** :doc:`M Layer v2 Documentation <m_layer_v2>`
    * **Examples:** :doc:`v2 Tutorials <tutorials_v2>` (covers both custom pipelines and P Layer usage).

* **The P Layer (Package):** Unlike v1 (which was purely for testing), the v2 Package layer provides a high-level, ``sklearn``-like interface along with the testing capabilities and tags registry. It wraps the M Layer to enable fast and easy training, prediction, and checkpointing without writing boilerplate PyTorch code. Simply pass a :py:class:`~pytorch_forecasting.data.timeseries.TimeSeries` object alongside your datamodule, model, and trainer configs to use ``model_pkg.fit()`` and ``model_pkg.predict()``.
    * **Learn more:** :doc:`P Layer Documentation <pkg_v2>`
    * **Examples:** :doc:`v2 Training and Inference Walkthrough </tutorials/ptf_V2_example>`.

Details and available models
-------------------------------

See the API documentation for further details on M layer and P layer and the list of the models:

.. toctree::
    :maxdepth: 2

    M Layer <m_layer_v2>
    P Layer <pkg_v2>
