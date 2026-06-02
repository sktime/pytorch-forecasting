M Layer (Model)
================

.. admonition::
   **Try the New API v2!**

   | You are viewing Documentation of v1 Models. A New API version is in development.
   | Explore the new architecture: :doc:`v2 Models <models_v2>` | :doc:`v2 API Reference <api_v2>`
   | **Caution: v2 is WIP and unstable. Not yet production-ready.**

.. currentmodule:: pytorch_forecasting

Model parameters very much depend on the dataset for which they are destined.

PyTorch Forecasting provides a ``.from_dataset()`` method for each model that
takes a :py:class:`~data.timeseries.TimeSeriesDataSet` and additional parameters
that cannot directly derived from the dataset such as, e.g. ``learning_rate`` or ``hidden_size``.

To tune models, `optuna <https://optuna.readthedocs.io/>`_ can be used. For example, tuning of the
:py:class:`~models.temporal_fusion_transformer.TemporalFusionTransformer`
is implemented by :py:func:`~models.temporal_fusion_transformer.tuning.optimize_hyperparameters`

Available Models
----------------
Here is an overview over the pros and cons of the implemented models:

.. model-overview-v1::

Usage
-----
PyTorch Forecasting provides a ``.from_dataset()`` method for each model that
takes a :py:class:`~data.timeseries.TimeSeriesDataSet` and additional parameters
that cannot directly derived from the dataset such as, e.g. ``learning_rate`` or ``hidden_size``.

One example of using :py:class:`~data.timeseries.TimeSeriesDataSet` and models is given below:

.. code-block:: python

    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
    from lightning.pytorch.tuner import Tuner
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

    # load data
    data = ...

    # define dataset
    max_encoder_length = 36
    max_prediction_length = 6
    training_cutoff = "YYYY-MM-DD"  # day for cutoff

    training = TimeSeriesDataSet(
        data[lambda x: x.date < training_cutoff],
        time_idx= ...,
        target= ...,
        # weight="weight",
        group_ids=[ ... ],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[ ... ],
        static_reals=[ ... ],
        time_varying_known_categoricals=[ ... ],
        time_varying_known_reals=[ ... ],
        time_varying_unknown_categoricals=[ ... ],
        time_varying_unknown_reals=[ ... ],
    )

    # create validation and training dataset
    validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)
    batch_size = 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

    # define trainer with early stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        gradient_clip_val=0.1,
        limit_train_batches=30,
        callbacks=[lr_logger, early_stop_callback],
    )

    # create the model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=2,
        reduce_on_plateau_patience=4
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
    res = Tuner(trainer).lr_find(
        tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()

    # fit the model
    trainer.fit(
        tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )

Implementing new architectures
-------------------------------

Please see the :ref:`Using custom data and implementing custom models <new-model-tutorial>` tutorial on how implement basic and more advanced models.

Every model should inherit from a base model in :py:mod:`~pytorch_forecasting.models.base`.

.. autoclass:: pytorch_forecasting.models.base._base_model.BaseModel
   :noindex:
   :members: __init__



Details and available models
-------------------------------

See the API documentation for further details on available models:

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api

    models.deepar.DeepAR
    models.mlp.DecoderMLP
    models.nbeats.NBeats
    models.nbeats.NBeatsKAN
    models.nhits.NHiTS
    models.rnn.RecurrentNetwork
    models.temporal_fusion_transformer.TemporalFusionTransformer
    models.tide.TiDEModel
    models.timexer.TimeXer
    models.xlstm.xLSTMTime
