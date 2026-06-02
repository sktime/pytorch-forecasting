Package
=======

.. admonition::
   **Try the New API v2!**

   | You are viewing Documentation of v1 Package Layer. A New API version is in development.
   | Explore the new architecture: :doc:`v2 Package Layer <pkg_v2>` | :doc:`v2 API Reference <api_v2>`
   | **Caution: v2 is WIP and unstable. Not yet production-ready.**

.. currentmodule:: pytorch_forecasting

The Package layer (denoted by the ``_pkg`` suffix) is primarily an internal structural component. It serves as a **container for model metadata and testing fixtures**.

If you are a standard user building forecasting models, you will typically interact directly with the core models (e.g., ``NBeats.from_dataset()``) and bypass this layer entirely. However, if you are contributing a new model to the PyTorch Forecasting library, you must implement a corresponding Package class.

Responsibilities of a V1 Package
--------------------------------

A V1 Package class inherits from :py:class:`~models.base._base_object._BasePtForecaster` and is strictly responsible for three things:

1. **Model Registry Tags (``_tags``):** A dictionary defining the specific capabilities of the model. This includes supported prediction types, whether it supports exogenous variables, multi-target forecasting, and its relative computational cost. These tags are used to dynamically generate the model overview tables in the documentation.
2. **Testing Fixtures:** Methods like ``get_base_test_params()`` and ``_get_test_dataloaders_from()`` generate standard, valid configurations and dataloaders. These ensure the model can be automatically tested within the Continuous Integration (CI) pipeline without requiring manual test scripts.

Anatomy of a V1 Package
-----------------------

Here is a complete example of a Package container using the ``NBeats`` model:

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

   models.base._base_object._BasePtForecaster
   models.deepar._deepar_pkg.DeepAR_pkg
   models.mlp._decodermlp_pkg.DecoderMLP_pkg
   models.nbeats._nbeats_pkg.NBeats_pkg
   models.nbeats._nbeatskan_pkg.NBeatsKAN_pkg
   models.nhits._nhits_pkg.NHiTS_pkg
   models.rnn._rnn_pkg.RecurrentNetwork_pkg
   models.temporal_fusion_transformer._tft_pkg.TemporalFusionTransformer_pkg
   models.tide._tide_pkg.TiDEModel_pkg
