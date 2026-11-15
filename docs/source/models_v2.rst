Models v2
==========

.. warning::
    Please note that the v2 modules are currently in active-development and is in beta right now, so please use this API with caution.
    See complete documentation for v2 API :doc:`here <api_v2>` and stable v1 documentation :doc:`here <api>`.

.. _models:

.. currentmodule:: pytorch_forecasting

The forecasting models in the V2 ecosystem are designed with a strict emphasis on modularity and separation of concerns. The architecture decouples algorithmic logic from data processing, ensuring that models act as pure, data-agnostic PyTorch Lightning instances.

Architecture
------------
The v2 models in ``pytorch-forecasting`` are separated into two distinct layers:

* **The M Layer (Model):** The core ``torch`` neural network implementation, inheriting from PyTorch Lightning's ``LightningModule``. Designed for experienced developers, this layer allows you to bypass the forecaster wrapper to build fully custom training, testing, and prediction pipelines.
    * **Learn more:** :doc:`M Layer v2 Documentation <m_layer_v2>`
    * **Examples:** :doc:`v2 Tutorials <tutorials_v2>` (covers both custom pipelines and forecaster usage).

* **The Forecaster Layer:** A high-level, ``sklearn``-like interface that wraps the M Layer. It handles training, prediction, checkpointing, and the associated data module, so you can train and forecast without writing boilerplate PyTorch Lightning code. Pass a :py:class:`~pytorch_forecasting.data.timeseries.TimeSeries` object to ``forecaster.fit()`` and ``forecaster.predict()``.
    * **Learn more:** :doc:`v2 API Reference <api_v2>`
    * **Examples:** :doc:`v2 Training and Inference Walkthrough </tutorials/ptf_V2_example>`.


Usage
------

1. Using the Forecaster layer. The forecaster builds its data module from the :py:class:`~pytorch_forecasting.data.timeseries.TimeSeries` and returns predictions as a :py:class:`~pytorch_forecasting.data.timeseries.TimeSeries`.

.. code-block:: python

    from pytorch_forecasting.data.timeseries import TimeSeries
    from pytorch_forecasting.metrics import MAE, SMAPE
    from pytorch_forecasting.models.temporal_fusion_transformer import TFTForecaster

    # 1. D1 Layer: Create the TimeSeries dataset
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

    # 2. Forecaster Layer: Configure and fit the model
    forecaster = TFTForecaster(
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
    forecaster.fit(dataset)

    # 3. Generate predictions
    preds = forecaster.predict(dataset, return_info=["index", "x", "y"])


2. Without using the Forecaster layer, here we use the M Layer with :py:class:`~pytorch_forecasting.data.data_module.EncoderDecoderTimeSeriesDataModule` and Lightning's trainer. You can use your own implementations of the data module and trainer for this workflow.

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


Details and available models
-------------------------------

See the API documentation for the M Layer and the Forecaster layer, and the list of the models:

.. toctree::
    :maxdepth: 2

    M Layer <m_layer_v2>
    API reference <api_v2>
