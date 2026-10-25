Models
======

.. warning::
    The v2 API is in beta and unstable. It may change without warning and is not
    ready for production use.

.. _models:

.. currentmodule:: pytorch_forecasting

Each model comes in two classes. The model class holds the network and is a
Lightning ``LightningModule``. It knows nothing about DataFrames and gets its
input shapes from a datamodule's ``metadata``. The forecaster class wraps the
model together with its datamodule and a trainer, and gives you ``fit`` and
``predict``.

* :doc:`Forecasters <forecaster>`: the quickest way to train and predict. Start
  here unless you need your own training loop.
* :doc:`Model layer <m_layer_v2>`: the model classes, for custom pipelines and for
  implementing new architectures.

The :doc:`model overview <model_list>` lists every model with its capabilities
and the datamodule it needs.


Using a forecaster
------------------

.. code-block:: python

    from lightning.pytorch import Trainer

    from pytorch_forecasting.data import TimeSeries
    from pytorch_forecasting.data.data_module import (
        EncoderDecoderTimeSeriesDataModule,
    )
    from pytorch_forecasting.metrics import MAE, SMAPE
    from pytorch_forecasting.models.temporal_fusion_transformer._tft_forecaster_v2 import (
        TFTForecaster,
    )

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
        trainer=Trainer(max_epochs=5, accelerator="auto"),
        datamodule=EncoderDecoderTimeSeriesDataModule(
            max_encoder_length=30,
            max_prediction_length=5,
            batch_size=32,
        ),
    )

    forecaster.fit(dataset)
    preds = forecaster.predict(dataset)


Using the model directly
------------------------

Without a forecaster, you build the datamodule with the data, pass its
``metadata`` to the model and train with your own ``Trainer``. Any Lightning
datamodule and trainer setup works here.

.. code-block:: python

    from lightning.pytorch import Trainer

    from pytorch_forecasting.data.data_module import (
        EncoderDecoderTimeSeriesDataModule,
    )
    from pytorch_forecasting.metrics import MAE, SMAPE
    from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import TFT

    data_module = EncoderDecoderTimeSeriesDataModule(
        time_series=dataset,
        max_encoder_length=30,
        max_prediction_length=5,
        batch_size=32,
    )

    model = TFT(
        loss=MAE(),
        logging_metrics=[MAE(), SMAPE()],
        optimizer="adam",
        optimizer_params={"lr": 1e-3},
        hidden_size=64,
        num_layers=2,
        attention_head_size=4,
        dropout=0.1,
        metadata=data_module.metadata,
    )

    trainer = Trainer(max_epochs=5, accelerator="auto")
    trainer.fit(model, data_module)


.. toctree::
    :maxdepth: 2

    Forecasters <forecaster>
    Model layer <m_layer_v2>
