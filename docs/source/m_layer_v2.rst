M Layer v2
===========

.. warning::
    Please note that the v2 modules are currently in active-development and is in beta right now, so please use this API with caution.
    See complete documentation for v2 API :doc:`here <api_v2>` and stable v1 documentation :doc:`here <api>`.

.. _m-layer-v2:

.. currentmodule:: pytorch_forecasting

The M layer contains the PyTorch Lightning model implementations. Each model
defines the forecasting architecture and the steps Lightning uses for training,
validation, testing, and prediction. It receives batches from a compatible
DataModule and does not own raw-data preparation.

Use a model's Forecaster for the higher-level ``fit`` and ``predict``
interface. To customize the Lightning workflow, create a compatible DataModule
and pass the M-layer model and DataModule to a Lightning ``Trainer`` directly.
The v2 workflow and its data and forecaster layers are described in
:doc:`Models v2 <models_v2>`.

Available models
----------------

The overview below lists the v2 models and their compatible data modules.

.. model-overview-v2::

Implementing a model
--------------------

See the `v2 extension templates
<https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v2>`_
for examples of the expected model and test structure. Model implementations
inherit from :class:`~pytorch_forecasting.models.base._base_model_v2.BaseModel`
or a specialized base model, such as
:class:`~pytorch_forecasting.models.base._tslib_base_model_v2.TslibBaseModel`.
The DataModule supplies metadata that models use to configure their input
layers. See :doc:`Data v2 <data_v2>` for the available DataModules and their
responsibilities.

Base model
----------

The base model provides the common Lightning interface for metrics, optimizer
and scheduler configuration, and training, evaluation, and prediction steps.
Architecture-specific code implements the model's forward pass.

.. autoclass:: pytorch_forecasting.models.base._base_model_v2.BaseModel
   :noindex:
   :members: __init__

API reference
-------------

See the detailed API documentation for the v2 base classes and model
implementations below.

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
   models.tsmixer._tsmixer_v2.TSMixer
   models.xlstm._xlstm_v2.xLSTMTime_v2
   models.mlp._decodermlp_v2.DecoderMLP_v2
   models.softs._softs_v2.SOFTS
   models.scinet._scinet_v2.SCINet_v2
   models.patch_tst._patch_tst_v2.PatchTST_v2
   models.frets._frets_v2.FreTS
   models.modern_tcn._modern_tcn_v2.ModernTCN
   models.units._units_v2.UniTS
   models.nhits._nhits_v2.NHiTS_v2
