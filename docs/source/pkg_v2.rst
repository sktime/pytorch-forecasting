Package (P) Layer v2
=====================

.. warning::
    Please note that the v2 modules are currently in active-development and is in beta right now, so please use this API with caution.
    See complete documentation for v2 API :doc:`here <api_v2>` and stable v1 documentation :doc:`here <api>`.

.. currentmodule:: pytorch_forecasting

The Package Layer is the defining feature of the V2 user experience. While the underlying D1, D2, and Model layers are strictly decoupled to maintain separation of concerns, the Package layer acts as the **Orchestrator**.

It wraps the PyTorch Lightning boilerplate and provides a streamlined, ``scikit-learn``-like interface. By passing standard Python dictionaries for configuration, the Package layer automatically handles the instantiation of the DataModule, extracts the necessary metadata, initializes the underlying forecasting model, and manages the training loop.

The Configuration Driven Workflow
---------------------------------

To use a V2 Package, you do not need to manually instantiate the DataModule or the Model. Instead, you provide three configuration dictionaries to the Package wrapper:

1. ``datamodule_cfg``: Parameters for the D2 layer (e.g., batch size, scalers, encoders, and maximum sequence lengths).
2. ``model_cfg``: Parameters for the core neural network (e.g., hidden sizes, dropout, loss metrics, and optimizers).
3. ``trainer_cfg``: Parameters for the PyTorch Lightning ``Trainer`` (e.g., max epochs, accelerator type, and logging configurations).

`sklearn` like Methods
-----------------------

The Package layer exposes two primary methods for interacting with the model lifecycle:

* ``fit()``: Initiates the training process. You can pass a raw D1 ``TimeSeries`` dataset (which the package will internally wrap in a D2 DataModule) or a pre-configured D2 ``LightningDataModule``.
* ``predict()``: Generates forecasts. Similar to ``fit()``, this accepts a D1 dataset, a D2 DataModule, or even a standard PyTorch ``DataLoader``. It also accepts a ``return_info`` parameter to easily append identifying columns (like time indices and series IDs) alongside the predictions.

All package classes inherit from :py:class:`~pytorch_forecasting.base._base_pkg.Base_pkg`

.. autoclass:: pytorch_forecasting.base._base_pkg.Base_pkg
   :noindex:
   :members: __init__


Code Example
------------

Here is how the configuration dictionaries and lifecycle methods come together using the Temporal Fusion Transformer package (``TFT_pkg_v2``):

.. code-block:: python

    from pytorch_forecasting.models.temporal_fusion_transformer._tft_pkg_v2 import TFT_pkg_v2
    from pytorch_forecasting.metrics import MAE, SMAPE
    from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer
    from sklearn.preprocessing import StandardScaler

    # Define Configurations
    datamodule_cfg = dict(
        max_encoder_length=30,
        max_prediction_length=1,
        batch_size=32,
    )

    model_cfg = dict(
        loss=MAE(),
        logging_metrics=[MAE(), SMAPE()],
        optimizer="adam",
        optimizer_params={"lr": 1e-3},
        hidden_size=64,
        num_layers=2,
    )

    trainer_cfg = dict(
        max_epochs=5,
        accelerator="auto",
        devices=1,
    )

    # Initialize the Package
    model_pkg = TFT_pkg_v2(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        datamodule_cfg=datamodule_cfg,
    )

    # Train the model
    # (Assuming `dataset` is a previously defined D1 TimeSeries object)
    model_pkg.fit(dataset)

    # Generate predictions
    preds = model_pkg.predict(dataset, return_info=["index", "x", "y"])


API Reference
-------------

See the detailed API documentation for the available V2 Package classes below:

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api

   models.temporal_fusion_transformer._tft_pkg_v2.TFT_pkg_v2
   models.dlinear._dlinear_pkg_v2.DLinear_pkg_v2
   models.samformer._samformer_v2_pkg.Samformer_pkg_v2
   models.tide._tide_dsipts._tide_v2_pkg.TIDE_pkg_v2
   models.timexer._timexer_pkg_v2.TimeXer_pkg_v2
