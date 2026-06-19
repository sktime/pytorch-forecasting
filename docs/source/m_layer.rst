M Layer (Model)
================

.. admonition::
   **Try the API v2 pre-release!**

   | You are viewing Documentation of v1 Models. A New API version 2 is in development.
   | Try it out before release: :doc:`v2 Models <models_v2>` | :doc:`v2 API Reference <api_v2>`
   | **Caution: v2 is WIP and unstable. Not yet production-ready.**

.. currentmodule:: pytorch_forecasting

Model parameters very much depend on the dataset for which they are destined.

PyTorch Forecasting provides a ``.from_dataset()`` method for each model that
takes a :py:class:`~data.timeseries.TimeSeriesDataSet` and additional parameters
that cannot directly derived from the dataset such as, e.g. ``learning_rate`` or ``hidden_size``.

To tune models, `optuna <https://optuna.readthedocs.io/>`_ can be used. For example, tuning of the
:py:class:`~models.temporal_fusion_transformer.TemporalFusionTransformer`
is implemented by :py:func:`~models.temporal_fusion_transformer.tuning.optimize_hyperparameters`

Available Models
----------------
Here is an overview over the pros and cons of the implemented models:

.. model-overview-v1::

Implementing new architectures
-------------------------------

Please see the :ref:`Using custom data and implementing custom models <new-model-tutorial>` tutorial and `extension templates <https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v1>`_ to understand how implement basic and more advanced models.

Every model should inherit from a base model in :py:mod:`~pytorch_forecasting.models.base`.

.. autoclass:: pytorch_forecasting.models.base._base_model.BaseModel
   :noindex:
   :members: __init__



Details and available models
-------------------------------

See the API documentation for further details on available models:

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api

    models.deepar.DeepAR
    models.mlp.DecoderMLP
    models.nbeats.NBeats
    models.nbeats.NBeatsKAN
    models.nhits.NHiTS
    models.rnn.RecurrentNetwork
    models.temporal_fusion_transformer.TemporalFusionTransformer
    models.tide.TiDEModel
    models.timexer.TimeXer
    models.xlstm.xLSTMTime
