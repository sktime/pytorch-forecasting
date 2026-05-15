Models
======

.. currentmodule:: pytorch_forecasting.models

This module contains neural network architectures for time series forecasting.
Models are organized by their type: forecasting models, base classes, and building blocks.

Forecasting Models
------------------

.. autosummary::
   :toctree: ../api

   TemporalFusionTransformer
   NBeats
   NBeatsKAN
   NHiTS
   DeepAR
   DLinear
   RecurrentNetwork
   TiDEModel
   TimeXer
   xLSTMTime
   Baseline
   DecoderMLP
   Samformer

Base Classes
------------

.. autosummary::
   :toctree: ../api

   BaseModel
   BaseModelWithCovariates
   AutoRegressiveBaseModel
   AutoRegressiveBaseModelWithCovariates

Building Blocks (NN)
--------------------

.. autosummary::
   :toctree: ../api

   get_rnn
   LSTM
   GRU
   MultiEmbedding
