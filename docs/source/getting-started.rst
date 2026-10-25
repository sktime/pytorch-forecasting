Getting started
===============

.. warning::
    The v2 API is in beta and unstable. It may change without warning and is not
    ready for production use.

.. _getting-started:


Installation
------------

.. _install:

Install ``pytorch-forecasting`` with

.. code-block:: bash

    pip install pytorch-forecasting

For a specific ``torch`` version or for the MQF2 loss, see the
:doc:`installation guide <installation>`.


Usage
-----

.. currentmodule:: pytorch_forecasting

Models train with `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_,
so CPUs, GPUs and several devices work without code changes. Training a model
takes four steps.

#. Wrap your DataFrame in a :py:class:`~data.timeseries._timeseries_v2.TimeSeries`
   and say which columns are the time index, target, groups and features.
#. Configure the datamodule the model needs, such as
   :py:class:`~data.data_module._encoder_decoder_data_module.EncoderDecoderTimeSeriesDataModule`.
   The :doc:`model overview <model_list>` lists the datamodule for each model.
#. Create the forecaster with the model's arguments, a Lightning ``Trainer`` and
   the datamodule.
#. Call ``fit`` with the ``TimeSeries``, then ``predict``.

Coming from v1? See :doc:`migration`.


Example
-------

.. code-block:: python

    from lightning.pytorch import Trainer

    from pytorch_forecasting.data import TimeSeries
    from pytorch_forecasting.data.data_module import (
        EncoderDecoderTimeSeriesDataModule,
    )
    from pytorch_forecasting.metrics import MAE
    from pytorch_forecasting.models.temporal_fusion_transformer._tft_forecaster_v2 import (
        TFTForecaster,
    )

    # data_df has one row per series and time step
    dataset = TimeSeries(
        data=data_df,
        time="time_idx",
        target="y",
        group=["series_id"],
        num=["x", "future_known_feature"],
        cat=["category"],
        known=["future_known_feature"],
        unknown=["x", "category"],
    )

    forecaster = TFTForecaster(
        hidden_size=32,
        attention_head_size=2,
        loss=MAE(),
        optimizer_params={"lr": 1e-3},
        trainer=Trainer(max_epochs=10, accelerator="auto"),
        datamodule=EncoderDecoderTimeSeriesDataModule(
            max_encoder_length=36,
            max_prediction_length=6,
            target_normalizer="auto",
            batch_size=64,
        ),
    )

    best_ckpt = forecaster.fit(dataset)
    preds = forecaster.predict(dataset, return_info=["y"])
    print(preds.to_pandas().head())

The :doc:`tutorials <tutorials_v2>` walk through complete examples.
