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
The v2 models in ``pytorch-forecasting`` are separated into two distinct sub-layers:

* **The M Layer (Model):** The core ``torch`` neural network implementation, inheriting from PyTorch Lightning's ``LightningModule``. Designed for experienced developers, this layer allows you to bypass the package wrapper to build fully custom training, testing, and prediction pipelines.
    * **Learn more:** :doc:`M Layer v2 Documentation <m_layer_v2>`
    * **Examples:** :doc:`v2 Tutorials <tutorials_v2>` (covers both custom pipelines and P Layer usage).

* **The Forecaster Layer:** Unlike v1 (where the package layer was purely internal for testing), the v2 Forecaster layer provides a high-level, ``sklearn``-like interface along with testing capabilities and tags registry. It wraps the M Layer and D2 DataModule to enable fast and easy training, prediction, and checkpointing without writing boilerplate PyTorch code. Simply configure model and training parameters to use ``forecaster.fit()`` and ``forecaster.predict()``.
    * **Learn more:** :doc:`Forecaster Layer Documentation <forecaster_v2>`
    * **Examples:** :doc:`v2 Training and Inference Walkthrough </tutorials/ptf_V2_example>`.


Usage
------

Because the Forecaster layer acts as a high-level orchestrator, you can directly configure model parameters and optionally supply configured DataModule and Trainer instances. See :doc:`Forecaster Layer Documentation <forecaster_v2>` for more info.

Here is a complete example of the V2 workflow using the Temporal Fusion Transformer (TFT):

1. Using :doc:`Forecaster Class <forecaster_v2>`:

.. code-block:: python

    from lightning.pytorch import Trainer
    from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
    from pytorch_forecasting.data.timeseries import TimeSeries
    from pytorch_forecasting.metrics import MAE, SMAPE
    from pytorch_forecasting.models.temporal_fusion_transformer import TFTForecaster

    # 1. D1 Layer: Create the TimeSeries dataset
    # This takes the raw pandas DataFrame and prepares the schema
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

    # 2. D2 Layer: Configure the DataModule
    datamodule = EncoderDecoderTimeSeriesDataModule(
        max_encoder_length=30,
        max_prediction_length=1,
        batch_size=32,
    )

    # 3. Trainer Configuration
    trainer = Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # 4. Forecaster Layer: Orchestration
    forecaster = TFTForecaster(
        hidden_size=64,
        num_layers=2,
        attention_head_size=4,
        dropout=0.1,
        loss=MAE(),
        logging_metrics=[MAE(), SMAPE()],
        optimizer="adam",
        optimizer_params={"lr": 1e-3},
        lr_scheduler="reduce_lr_on_plateau",
        lr_scheduler_params={"mode": "min", "factor": 0.1, "patience": 10},
        trainer=trainer,
        datamodule=datamodule,
    )

    # Fit the model (You can pass the D1 dataset or a D2 DataModule here)
    forecaster.fit(dataset)

    # Generate predictions (returns a TimeSeries object)
    preds = forecaster.predict(dataset, return_info=["index", "x", "y"])


2. Without using ``Forecaster`` class, here we are using :py:class:`~pytorch_forecasting.data.data_module.EncoderDecoderTimeSeriesDataModule` and ``Lightning``'s trainer, but you can use your own implementations of the datamodule and trainer for this workflow.

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

See the API documentation for further details on M layer and Forecaster layer and the list of the models:

.. toctree::
    :maxdepth: 2

    M Layer <m_layer_v2>
    Forecaster Layer <forecaster_v2>
