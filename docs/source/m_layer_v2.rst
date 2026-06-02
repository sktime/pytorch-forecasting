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


Usage
------

Because the Package layer acts as a high-level orchestrator, the workflow for setting up and training a model relies on configuration dictionaries. You define the data and the configurations, and pass them directly to the package class. See :doc:`Package classes Documentation <pkg>` for more info.

Here is a complete example of the V2 workflow using the Temporal Fusion Transformer (TFT):

1. Using :doc:`Package Class <pkg_v2>`:

.. code-block:: python

    from pytorch_forecasting.models.temporal_fusion_transformer._tft_pkg_v2 import TFT_pkg_v2
    from pytorch_forecasting.data.timeseries._timeseries_v2 import TimeSeries
    from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer
    from sklearn.preprocessing import StandardScaler
    from pytorch_forecasting.metrics import MAE, SMAPE

    # 1. D1 Layer: Create the TimeSeries dataset
    # This takes the raw pandas DataFrame and prepares the tensor extraction
    dataset = TimeSeries(
        data=data_df,
        time="time_idx",
        target="y",
        group=["series_id"],
        num=["x", "future_known_feature", "static_feature"],
        cat=["category", "static_feature_cat"],
        known=["future_known_feature"],
        unknown=["x", "category"],
        static=["static_feature", "static_feature_cat"],
    )

    # 2. D2 Layer Configuration
    datamodule_cfg = dict(
        max_encoder_length=30,
        max_prediction_length=1,
        batch_size=32,
    )

    # 3. Model Configuration
    model_cfg = dict(
        loss=MAE(),
        logging_metrics=[MAE(), SMAPE()],
        optimizer="adam",
        optimizer_params={"lr": 1e-3},
        lr_scheduler="reduce_lr_on_plateau",
        lr_scheduler_params={"mode": "min", "factor": 0.1, "patience": 10},
        hidden_size=64,
        num_layers=2,
        attention_head_size=4,
        dropout=0.1,
    )

    # 4. Trainer Configuration
    trainer_cfg = dict(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # 5. Package Layer: Orchestration
    model_pkg = TFT_pkg_v2(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        datamodule_cfg=datamodule_cfg,
    )

    # Fit the model (You can pass the D1 dataset or a D2 DataModule here)
    model_pkg.fit(dataset)

    # Generate predictions (You can also pass a DataModule or Dataloader here)
    preds = model_pkg.predict(dataset, return_info=["index", "x", "y"])


2. Without using ``Package`` class, here we are using ``EncoderDecoderDataModule`` and ``Lightning``'s trainer, but you can use your own implementations of the datamodule and trainer for this workflow.

.. code-block:: python

    from pytorch_forecasting.data.timeseries import TimeSeries
    from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
    from pytorch_forecasting.metrics import MAE, SMAPE
    from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import TFT
    from lightning.pytorch import Trainer

    # create `TimeSeries` dataset that returns the raw data in terms of tensors
    dataset = TimeSeries(
        data=data_df,
        time="time_idx",
        target="y",
        group=["series_id"],
        num=["x", "future_known_feature", "static_feature"],
        cat=["category", "static_feature_cat"],
        known=["future_known_feature"],
        unknown=["x", "category"],
        static=["static_feature", "static_feature_cat"],
    )

    # create the `data_module` that handles the dataloaders and preprocessing
    data_module = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=dataset,
        max_encoder_length=30,
        max_prediction_length=1,
        batch_size=32,
    )

    # Initialise the Model
    model = TFT(
        loss=MAE(),
        logging_metrics=[MAE(), SMAPE()],
        optimizer="adam",
        optimizer_params={"lr": 1e-3},
        lr_scheduler="reduce_lr_on_plateau",
        lr_scheduler_params={"mode": "min", "factor": 0.1, "patience": 10},
        hidden_size=64,
        num_layers=2,
        attention_head_size=4,
        dropout=0.1,
        metadata=data_module.metadata,  # pass the metadata from the datamodule to the model
        # to initialise important params like `encoder_cont` etc
    )

    # Train the model
    trainer = Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)

Implementing new architectures
-------------------------------

Please see the Extension Templates to understand the process and design of the implementations.

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
