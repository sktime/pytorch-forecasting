Forecaster Layer v2
===================

.. warning::
    Please note that the v2 modules are currently in active-development and is in beta right now, so please use this API with caution.
    See complete documentation for v2 API :doc:`here <api_v2>` and stable v1 documentation :doc:`here <api>`.



Introduction
------------

The Forecaster Layer provides a high-level interface for training and
generating predictions with forecasting models.

It builds on the lower-level v2 components, including the :class:`~pytorch_forecasting.data.TimeSeries`
data structure and DataModule. Instead of manually managing the complete
training and prediction workflow, users can interact with a Forecaster
through a common interface.

All v2 Forecaster implementations inherit from
:class:`~pytorch_forecasting.models.base.BaseForecaster`.

BaseForecaster
--------------

:class:`BaseForecaster` provides the common functionality shared by
the v2 forecasting models. It manages the interaction between the
DataModule, the underlying forecasting model, and the PyTorch Lightning
Trainer.

The main methods are:

* ``fit()``: Prepares the data, initializes the forecasting model, and
  trains it using the configured Trainer.
* ``predict()``: Generates predictions for new data using the fitted
  model.

This common interface allows different forecasting architectures to
follow a consistent training and prediction workflow.

.. autoclass:: pytorch_forecasting.models.base.BaseForecaster
   :members: __init__, fit, predict
   :show-inheritance:

Forecaster Workflow
-------------------

A Forecaster provides a consistent workflow for training and generating
predictions using a :class:`~pytorch_forecasting.data.TimeSeries` object.

* **Training:** Pass a :class:`~pytorch_forecasting.data.TimeSeries` object to ``fit()``. The Forecaster
  attaches the data to its configured DataModule using ``with_data()``,
  initializes the model, and trains it.
* **Prediction:** Pass new :class:`~pytorch_forecasting.data.TimeSeries` data to ``predict()``. The fitted
  DataModule reuses its preprocessing transformations on the new data.
* **Prediction output:** In ``prediction`` and ``quantiles`` modes,
  predictions are returned as a :class:`~pytorch_forecasting.data.TimeSeries` object. Raw mode returns
  the underlying model output dictionary.

For tabular results, users can convert the returned :class:`~pytorch_forecasting.data.TimeSeries` object to a pandas DataFrame.
Forecaster Implementations
--------------------------

The following forecasting models provide the common
:class:`~pytorch_forecasting.models.base.BaseForecaster` interface:

.. autosummary::
   :toctree: api

   pytorch_forecasting.models.dlinear.DLinearForecaster
   pytorch_forecasting.models.frets.FreTSForecaster
   pytorch_forecasting.models.mlp.DecoderMLPForecaster
   pytorch_forecasting.models.modern_tcn.ModernTCNForecaster
   pytorch_forecasting.models.nhits.NHiTSForecaster
   pytorch_forecasting.models.patch_tst.PatchTSTForecaster
   pytorch_forecasting.models.samformer.SamformerForecaster
   pytorch_forecasting.models.scinet.SCINetForecaster
   pytorch_forecasting.models.softs.SOFTSForecaster
   pytorch_forecasting.models.temporal_fusion_transformer.TFTForecaster
   pytorch_forecasting.models.tide.TIDEForecaster
   pytorch_forecasting.models.timexer.TimeXerForecaster
   pytorch_forecasting.models.tsmixer.TSMixerForecaster
   pytorch_forecasting.models.units.UniTSForecaster
   pytorch_forecasting.models.xlstm.xLSTMTimeForecaster
