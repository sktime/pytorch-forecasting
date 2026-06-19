M Layer v2
===========

.. warning::
    Please note that the v2 modules are currently in active-development and is in beta right now, so please use this API with caution.
    See complete documentation for v2 API :doc:`here <api_v2>` and stable v1 documentation :doc:`here <api>`.

.. _models:

.. currentmodule:: pytorch_forecasting

The forecasting models in the V2 ecosystem are designed with a strict emphasis on modularity and separation of concerns. The architecture decouples algorithmic logic from data processing, ensuring that models act as pure, data-agnostic PyTorch Lightning instances.

Available Models
----------------

Below is a summary of the forecasting models currently implemented and supported in the new API.

.. model-overview-v2::

Implementing new architectures
-------------------------------

Please see the `Extension Templates <https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v2>`_ to understand the process and design of the implementations.

Every model should inherit from a base model in :py:mod:`~pytorch_forecasting.models.base._base_model_v2`.

.. autoclass:: pytorch_forecasting.models.base._base_model_v2.BaseModel
   :noindex:
   :members: __init__


API Reference
-------------

See the detailed API documentation for the V2 base classes and specific model implementations below:

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api

   models.base._base_model_v2.BaseModel
   models.base._tslib_base_model_v2.TslibBaseModel
   models.temporal_fusion_transformer._tft_v2.TFT
   models.dlinear._dlinear_v2.DLinear
   models.samformer._samformer_v2.Samformer
   models.tide._tide_dsipts._tide_v2.TIDE
   models.timexer._timexer_v2.TimeXer
