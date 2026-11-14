Forecaster Layer v2
====================

.. warning::
    Please note that the v2 modules are currently in active-development and is in beta right now, so please use this API with caution.
    See complete documentation for v2 API :doc:`here <api_v2>` and stable v1 documentation :doc:`here <api>`.

.. currentmodule:: pytorch_forecasting

The Forecaster Layer is the defining feature of the V2 user experience. While the underlying D1, D2, and Model layers are strictly decoupled to maintain separation of concerns, the Forecaster layer acts as the **Orchestrator**.

It wraps the PyTorch Lightning boilerplate and provides a streamlined, ``scikit-learn``-like interface. By passing configuration parameters directly to the Forecaster wrapper (or optionally providing pre-configured DataModule and Trainer instances), the Forecaster layer automatically handles data binding, initializes the underlying forecasting model, and manages the training and prediction loops.

Direct Configuration Workflow
-----------------------------

To use a V2 Forecaster, you configure model parameters directly during initialization. You can also optionally supply:

1. ``datamodule``: An un-fitted or data-less D2 DataModule (e.g. :py:class:`~pytorch_forecasting.data.data_module.EncoderDecoderTimeSeriesDataModule` or :py:class:`~pytorch_forecasting.data.data_module.TslibDataModule`). If omitted, the default datamodule class for the forecaster is instantiated.
2. ``trainer``: A configured PyTorch Lightning :py:class:`~lightning.pytorch.Trainer` (e.g., max epochs, accelerator type, and callbacks). If omitted, a default ``Trainer()`` is used.

`sklearn` like Methods
----------------------

The Forecaster layer exposes two primary methods for interacting with the model lifecycle:

* ``fit()``: Initiates the training process. You can pass a raw D1 ``TimeSeries`` dataset (which the forecaster attaches to the datamodule with ``with_data()``) or a pre-configured D2 ``LightningDataModule``.
* ``predict()``: Generates forecasts as a ``TimeSeries`` object. Similar to ``fit()``, this accepts a D1 dataset, a D2 DataModule, or a DataLoader. It also accepts a ``return_info`` parameter to easily append identifying columns (like time indices and series IDs) alongside the predictions.

All forecaster classes inherit from :py:class:`~pytorch_forecasting.models.base._base_forecaster.BaseForecaster`

.. autoclass:: pytorch_forecasting.models.base._base_forecaster.BaseForecaster
   :noindex:
   :members: __init__


Code Example
------------

Here is how the Forecaster workflow comes together using the Temporal Fusion Transformer (``TFTForecaster``):

.. code-block:: python

    from lightning.pytorch import Trainer
    from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
    from pytorch_forecasting.metrics import MAE, SMAPE
    from pytorch_forecasting.models.temporal_fusion_transformer import TFTForecaster

    # Configure the DataModule
    datamodule = EncoderDecoderTimeSeriesDataModule(
        max_encoder_length=30,
        max_prediction_length=1,
        batch_size=32,
    )

    # Configure the Trainer
    trainer = Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
    )

    # Initialize the Forecaster
    forecaster = TFTForecaster(
        hidden_size=64,
        num_layers=2,
        attention_head_size=4,
        dropout=0.1,
        loss=MAE(),
        logging_metrics=[MAE(), SMAPE()],
        optimizer="adam",
        optimizer_params={"lr": 1e-3},
        trainer=trainer,
        datamodule=datamodule,
    )

    # Train the model
    # (Assuming `dataset` is a previously defined D1 TimeSeries object)
    forecaster.fit(dataset)

    # Generate predictions
    preds = forecaster.predict(dataset, return_info=["index", "x", "y"])


API Reference
-------------

See the detailed API documentation for the available V2 Forecaster classes below:

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api

   models.temporal_fusion_transformer._tft_forecaster_v2.TFTForecaster
   models.dlinear._dlinear_forecaster_v2.DLinearForecaster
   models.samformer._samformer_forecaster_v2.SamformerForecaster
   models.tide._tide_dsipts._tide_forecaster_v2.TIDEForecaster
   models.timexer._timexer_forecaster_v2.TimeXerForecaster
   models.xlstm._xlstm_forecaster_v2.xLSTMTimeForecaster
   models.mlp._decodermlp_forecaster_v2.DecoderMLPForecaster
   models.softs._softs_forecaster_v2.SOFTSForecaster
   models.scinet._scinet_forecaster_v2.SCINetForecaster
   models.patch_tst._patch_tst_forecaster_v2.PatchTSTForecaster
   models.frets._frets_forecaster_v2.FreTSForecaster
