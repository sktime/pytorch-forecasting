Model layer
===========

.. warning::
    The v2 API is in beta and unstable. It may change without warning and is not
    ready for production use.

.. currentmodule:: pytorch_forecasting

The model layer holds the networks. Each model is a Lightning ``LightningModule``
with ``forward(x: dict) -> dict``. It is built from the ``metadata`` of its
datamodule and never reads raw data itself. The :doc:`model overview <model_list>`
lists the available models.

Most users train through a :doc:`forecaster <forecaster>`. Use the model classes
directly when you need your own training loop, as shown in :doc:`models_v2`.


Implementing new architectures
------------------------------

Every model inherits from a base model in
:py:mod:`~pytorch_forecasting.models.base._base_model_v2`. Models built for
:py:class:`~data.data_module._tslib_data_module.TslibDataModule` inherit from
:py:class:`~models.base._tslib_base_model_v2.TslibBaseModel` instead. Each model
also needs a forecaster class, see :doc:`forecaster`.

.. autoclass:: pytorch_forecasting.models.base._base_model_v2.BaseModel
   :noindex:
   :members: __init__


API Reference
-------------

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
   models.xlstm._xlstm_v2.xLSTMTime_v2
   models.mlp._decodermlp_v2.DecoderMLP_v2
   models.softs._softs_v2.SOFTS
   models.scinet._scinet_v2.SCINet_v2
   models.patch_tst._patch_tst_v2.PatchTST_v2
   models.frets._frets_v2.FreTS
