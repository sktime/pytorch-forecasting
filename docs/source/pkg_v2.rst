Forecaster (F) Layer v2
========================

.. warning::
    Please note that the v2 modules are currently in active-development and is in beta right now, so please use this API with caution.
    See complete documentation for v2 API :doc:`here <api_v2>` and stable v1 documentation :doc:`here <api>`.

.. currentmodule:: pytorch_forecasting

The Forecaster layer is the defining feature of the V2 user experience. While the underlying D1, D2, and Model layers are strictly decoupled to maintain separation of concerns, the Forecaster layer acts as the **Orchestrator**.

It wraps the PyTorch Lightning boilerplate and provides a streamlined, ``scikit-learn``-like interface. The forecaster builds the model once the input shapes are known - they are read from the metadata of the datamodule - and manages the training loop, the checkpointing and the prediction pipeline.

.. note::
    In earlier pre-releases this layer was called the *Package* (P) layer, and its
    classes were named ``<Model>_pkg_v2``. They are now named ``<Model>Forecaster``
    and take model parameters directly instead of ``model_cfg``/``trainer_cfg``/``datamodule_cfg``
    dictionaries.

The Forecaster Workflow
-----------------------

To use a V2 forecaster, you do not need to manually instantiate the model. You pass:

1. The model's own parameters (e.g. hidden sizes, dropout, loss metrics, and optimizers) directly to the forecaster - they mirror the parameters of the underlying M layer model.
2. ``datamodule``: a configured but data-less D2 datamodule (e.g. batch size, scalers, encoders, and maximum sequence lengths). ``fit`` attaches the data to it. Defaults to the model's default datamodule class.
3. ``trainer``: a ``lightning.Trainer`` (e.g. max epochs, accelerator type, and logging configurations). Each ``fit`` trains on a new trainer with this configuration. Defaults to ``Trainer()``.

`sklearn` like Methods
-----------------------

The Forecaster layer exposes two primary methods for interacting with the model lifecycle:

* ``fit()``: Initiates the training process. It takes a D1 :py:class:`~pytorch_forecasting.data.timeseries._timeseries_v2.TimeSeries` dataset, which the datamodule splits into training and validation data. By default it saves the best checkpoint together with the datamodule config and the metadata, so the forecaster can later be restored with ``ckpt_path``.
* ``predict()``: Generates forecasts for a ``TimeSeries``. It returns a ``TimeSeries`` of predictions for ``mode="prediction"`` and ``"quantiles"``, or the model's raw output dict for ``mode="raw"``. It also accepts a ``return_info`` parameter to easily append extra model outputs (like ``"index"``, ``"x"`` and ``"y"``) alongside the predictions.

All forecaster classes inherit from :py:class:`~pytorch_forecasting.models.base._base_forecaster.BaseForecaster`

.. autoclass:: pytorch_forecasting.models.base._base_forecaster.BaseForecaster
   :noindex:
   :members: __init__, fit, predict


Code Example
------------

Here is how the model parameters, the datamodule, the trainer and the lifecycle methods come together using the Temporal Fusion Transformer forecaster (``TFTForecaster``):

.. code-block:: python

    from lightning.pytorch import Trainer
    from sklearn.preprocessing import StandardScaler

    from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
    from pytorch_forecasting.data.encoders import TorchNormalizer
    from pytorch_forecasting.metrics import MAE, SMAPE
    from pytorch_forecasting.models.temporal_fusion_transformer import TFTForecaster

    # Configure the D2 layer - no data is passed here, `fit` attaches it
    datamodule = EncoderDecoderTimeSeriesDataModule(
        max_encoder_length=30,
        max_prediction_length=1,
        batch_size=32,
        scalers={"x": StandardScaler()},
        target_normalizer=TorchNormalizer(),
    )

    # Initialize the Forecaster - model parameters are passed directly
    forecaster = TFTForecaster(
        hidden_size=64,
        num_layers=2,
        loss=MAE(),
        logging_metrics=[MAE(), SMAPE()],
        optimizer="adam",
        optimizer_params={"lr": 1e-3},
        datamodule=datamodule,
        trainer=Trainer(max_epochs=5, accelerator="auto", devices=1),
    )

    # Train the model
    # (Assuming `dataset` is a previously defined D1 TimeSeries object)
    forecaster.fit(dataset)

    # Generate predictions
    preds = forecaster.predict(dataset, return_info=["index", "x", "y"])

A fitted forecaster can be restored from the checkpoint written by ``fit``:

.. code-block:: python

    forecaster = TFTForecaster(ckpt_path="checkpoints/best-epoch=4-step=100.ckpt")
    preds = forecaster.predict(dataset)


API Reference
-------------

See the detailed API documentation for the available V2 forecaster classes below:

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api

   models.temporal_fusion_transformer._tft_forecaster_v2.TFTForecaster
   models.dlinear._dlinear_forecaster_v2.DLinearForecaster
   models.samformer._samformer_forecaster_v2.SamformerForecaster
   models.tide._tide_dsipts._tide_forecaster_v2.TIDEForecaster
   models.timexer._timexer_forecaster_v2.TimeXerForecaster
   models.tsmixer._tsmixer_forecaster_v2.TSMixerForecaster
   models.xlstm._xlstm_forecaster_v2.xLSTMTimeForecaster
   models.mlp._decodermlp_forecaster_v2.DecoderMLPForecaster
   models.softs._softs_forecaster_v2.SOFTSForecaster
   models.scinet._scinet_forecaster_v2.SCINetForecaster
   models.patch_tst._patch_tst_forecaster_v2.PatchTSTForecaster
   models.frets._frets_forecaster_v2.FreTSForecaster
   models.modern_tcn._modern_tcn_forecaster_v2.ModernTCNForecaster
   models.units._units_forecaster_v2.UniTSForecaster
   models.nhits._nhits_forecaster_v2.NHiTSForecaster
