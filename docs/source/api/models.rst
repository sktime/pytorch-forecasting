Models
======

.. currentmodule:: pytorch_forecasting.models

This module contains neural network architectures for time series forecasting.
Models are organized by their type: standard architectures, ensemble methods, and experimental versions.

Neural Networks
---------------

.. autosummary::
   :toctree: ../api

   TemporalFusionTransformer
   NBeats
   DeepAR
   Transformer
   LSTM

Package Containers
-------------------

.. autosummary::
   :toctree: ../api

   BaseModel
   MultiHorizonMetricMixin

V2 Objects (Beta)
------------------

.. note::
   The following models are experimental and part of the v2 API.
   Expect breaking changes in future releases.

.. autosummary::
   :toctree: ../api

   DeepARv2
   TemporalFusionTransformerv2
