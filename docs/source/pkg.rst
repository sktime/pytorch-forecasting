P Layer (Package)
==================

.. admonition::
   **Try the API v2 pre-release!**

   | You are viewing Documentation of v1 Package Layer. A New API version 2 is in development.
   | Try it out before release: :doc:`v2 Package Layer <pkg_v2>` | :doc:`v2 API Reference <api_v2>`
   | **Caution: v2 is WIP and unstable. Not yet production-ready.**

.. currentmodule:: pytorch_forecasting

The Package layer (denoted by the ``_pkg`` suffix) is a private, internal structural component of the v1 architecture. It serves as a centralized **container for model metadata, capability tags, dependency management, and testing fixtures**.

If you are a standard user building forecasting models, you will typically interact directly with the core models (e.g., ``NBeats.from_dataset()``) and bypass this layer entirely. However, if you are contributing a new model to the PyTorch Forecasting library, you must implement a corresponding Package class (see the extension templates `here <https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v1>`_ for more info on how to implement this class).

Responsibilities of a v1 Package
--------------------------------

A v1 Package class inherits from :py:class:`~models.base._base_object._BasePtForecaster` and is strictly responsible for managing the model's ecosystem integration:

1. **Model Linkage** (``get_cls``): It acts as a lazy-loading proxy that returns the actual PyTorch Lightning model class without triggering heavy imports across the framework.
2. **Metadata & Capability Tags** (``_tags``): A comprehensive dictionary defining the model's structural profile. This includes the target data types, supported prediction types (e.g., ``point``, ``quantile``), exogenous variable support, multivariate capabilities, computational intensity, and author attribution. These tags dynamically populate the model overview tables and to understand the properties of the models.
3. **Dependency Management:** Through the ``python_dependencies`` tag, the package container declares any specific external packages required by the model, allowing the framework to manage optional imports gracefully.
4. **Testing Fixtures:** Methods like ``get_base_test_params()`` and ``_get_test_dataloaders_from()`` generate standard, valid configurations and train/validation dataloaders. These ensure the model can be seamlessly validated within the Continuous Integration (CI) pipeline.

Anatomy of a v1 Package
-----------------------

By convention, the package container must be a private file (e.g., ``_nbeats_pkg.py``) and its class name must exactly match the model name with a ``_pkg`` suffix (e.g., ``NBeats_pkg``).

Below is the auto-generated documentation for the ``NBeats`` package. To see exactly how the tags and testing fixtures are implemented in the code, click the **[source]** button next to the class name:

.. autoclass:: pytorch_forecasting.models.nbeats._nbeats_pkg.NBeats_pkg
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:

API Reference
-------------

See the detailed API documentation for the V1 Package classes below:

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api

   models.deepar._deepar_pkg.DeepAR_pkg
   models.mlp._decodermlp_pkg.DecoderMLP_pkg
   models.nbeats._nbeats_pkg.NBeats_pkg
   models.nbeats._nbeatskan_pkg.NBeatsKAN_pkg
   models.nhits._nhits_pkg.NHiTS_pkg
   models.rnn._rnn_pkg.RecurrentNetwork_pkg
   models.temporal_fusion_transformer._tft_pkg.TemporalFusionTransformer_pkg
   models.tide._tide_pkg.TiDEModel_pkg
