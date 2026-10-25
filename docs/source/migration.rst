Migrating from v1
=================

.. warning::
    The v2 API is in beta and unstable. It may change without warning and is not
    ready for production use.

.. currentmodule:: pytorch_forecasting

This page is for code written against v1, which builds a ``TimeSeriesDataSet``
and calls ``Model.from_dataset``. The v1 classes are still in the package, so
existing code keeps running while you port it.

In v1, ``TimeSeriesDataSet`` held the data, encoded and scaled it, cut the
windows and built the dataloaders. In v2 that work is split.
:py:class:`~data.timeseries._timeseries_v2.TimeSeries` holds the data, a
datamodule does the preprocessing and windowing, and a forecaster builds the model
and trains it. See :doc:`api_v2` for how the layers fit together.


The same task in both APIs
--------------------------

A Temporal Fusion Transformer in v1.

.. code-block:: python

    import lightning.pytorch as pl

    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import MAE

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="y",
        group_ids=["series_id"],
        max_encoder_length=30,
        max_prediction_length=5,
        static_categoricals=["static_feature_cat"],
        static_reals=["static_feature"],
        time_varying_known_reals=["future_known_feature"],
        time_varying_unknown_categoricals=["category"],
        time_varying_unknown_reals=["x", "y"],
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, data, min_prediction_idx=training_cutoff + 1
    )
    train_dl = training.to_dataloader(train=True, batch_size=32)
    val_dl = validation.to_dataloader(train=False, batch_size=32)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        loss=MAE(),
        learning_rate=1e-3,
        reduce_on_plateau_patience=4,
    )
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
    predictions = tft.predict(val_dl)

The same model in v2.

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

    dataset = TimeSeries(
        data=data,
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
        attention_head_size=4,
        dropout=0.1,
        loss=MAE(),
        optimizer_params={"lr": 1e-3},
        lr_scheduler="reduce_lr_on_plateau",
        lr_scheduler_params={"patience": 4},
        trainer=Trainer(max_epochs=5),
        datamodule=EncoderDecoderTimeSeriesDataModule(
            max_encoder_length=30,
            max_prediction_length=5,
            target_normalizer="auto",
            batch_size=32,
        ),
    )
    forecaster.fit(dataset)
    predictions = forecaster.predict(dataset)


Where each v1 argument goes
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - v1
     - v2
   * - ``TimeSeriesDataSet(time_idx=...)``
     - ``TimeSeries(time=...)``
   * - ``group_ids``
     - ``TimeSeries(group=...)``
   * - ``target``, ``weight``
     - ``TimeSeries(target=..., weight=...)``
   * - ``static_categoricals``, ``static_reals``
     - ``TimeSeries(static=...)``, with each column also in ``cat`` or ``num``
   * - ``time_varying_known_categoricals``, ``time_varying_known_reals``
     - ``TimeSeries(known=...)``, with each column also in ``cat`` or ``num``
   * - ``time_varying_unknown_categoricals``, ``time_varying_unknown_reals``
     - ``TimeSeries(unknown=...)``, with each column also in ``cat`` or ``num``
   * - ``max_encoder_length``, ``max_prediction_length``
     - datamodule arguments of the same name. ``TslibDataModule`` calls them
       ``context_length`` and ``prediction_length``.
   * - ``target_normalizer``, ``scalers``
     - datamodule arguments of the same name
   * - ``categorical_encoders``
     - accepted by the datamodule but not used for now
   * - ``TimeSeriesDataSet.from_dataset(training, data, ...)``
     - datamodule ``train_val_test_split``
   * - ``dataset.to_dataloader(...)``
     - the datamodule's ``train_dataloader()`` and related methods, called by the
       forecaster
   * - ``Model.from_dataset(dataset, **kwargs)``
     - ``ModelForecaster(**kwargs)``, or ``Model(metadata=datamodule.metadata, **kwargs)``
   * - ``learning_rate``
     - ``optimizer_params={"lr": ...}``
   * - ``reduce_on_plateau_patience``
     - ``lr_scheduler="reduce_lr_on_plateau"`` with
       ``lr_scheduler_params={"patience": ...}``
   * - ``trainer.fit(model, train_dl, val_dl)``
     - ``forecaster.fit(dataset)``
   * - ``model.predict(dataloader)``
     - ``forecaster.predict(dataset)``
   * - ``Model.load_from_checkpoint(path)``
     - ``ModelForecaster(ckpt_path=path)``


Behaviour that differs
----------------------

Validation data is chosen differently. v1 code usually holds out the last time
steps of every series with ``from_dataset``. The v2 datamodules assign whole
series at random to train, validation and test, and never cut a series in time.

The target is not scaled by default in ``EncoderDecoderTimeSeriesDataModule``. v1
uses ``target_normalizer="auto"``. In v2 the default is ``None``, so pass
``"auto"`` to keep the v1 behaviour. ``TslibDataModule`` does not scale the data
for now.

Categorical columns must hold integer codes. v1 encodes string categories for
you, while v2 does not encode them yet.

Predictions come back as a ``TimeSeries`` with one row per forecast step. Call
``to_pandas()`` to get a DataFrame. The ``_series`` column is the position of the
series in the input, not its group id. Values are in the scale of the fitted
target normalizer, because the inverse transform is not applied yet. v1 returns
predictions in the original scale.

v2 models have no ``plot_prediction``, ``interpret_output`` or learning rate
finder yet.


Model availability
------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - v1 model
     - v2 forecaster
   * - ``TemporalFusionTransformer``
     - ``TFTForecaster``
   * - ``TiDEModel``
     - ``TIDEForecaster``, a separate implementation
   * - ``DecoderMLP``
     - ``DecoderMLPForecaster``
   * - ``xLSTMTime``
     - ``xLSTMTimeForecaster``
   * - ``TimeXer``
     - ``TimeXerForecaster``
   * - ``PatchTST``
     - ``PatchTSTForecaster``
   * - ``NBeats``, ``NBeatsKAN``, ``NHiTS``, ``DeepAR``, ``RecurrentNetwork``,
       ``Baseline``
     - not available in v2 yet

``DLinearForecaster``, ``SamformerForecaster``, ``SOFTSForecaster``,
``SCINetForecaster`` and ``FreTSForecaster`` have no v1 counterpart. See the
:doc:`model overview <model_list>` for all of them.
