Forecasters
===========

.. warning::
    The v2 API is in beta and unstable. It may change without warning and is not
    ready for production use.

.. currentmodule:: pytorch_forecasting

A forecaster is the class most users train and predict with. It pairs one model
with the datamodule that model needs, builds both, trains through a Lightning
``Trainer`` and returns predictions as a
:py:class:`~data.timeseries._timeseries_v2.TimeSeries`. Every forecaster subclasses
:py:class:`~models.base._base_forecaster.BaseForecaster`, and is named after its
model, for example ``TFTForecaster`` for ``TFT``.

The model and datamodule underneath are ordinary Lightning classes. To write your
own training loop, use them directly as shown in :doc:`models_v2`.


Creating a forecaster
---------------------

Model arguments such as ``hidden_size``, ``loss`` or ``optimizer`` go straight to
the forecaster's constructor. Every forecaster also takes three shared arguments.

* ``trainer``: a ``lightning.pytorch.Trainer``. Each call to ``fit`` trains on a
  fresh trainer with the same settings, so ``fit`` can be called more than once.
  Defaults to ``Trainer()``.
* ``datamodule``: a datamodule built without data, see :doc:`data_v2`. Defaults to
  the forecaster's datamodule class with its default settings.
* ``ckpt_path``: a checkpoint to load a trained model from, see
  `Checkpoints`_.

The model itself is built inside ``fit``, because its input shapes come from the
datamodule's metadata. Until then ``forecaster.model`` is ``None``.

.. code-block:: python

    from lightning.pytorch import Trainer

    from pytorch_forecasting.data.data_module import (
        EncoderDecoderTimeSeriesDataModule,
    )
    from pytorch_forecasting.metrics import MAE, SMAPE
    from pytorch_forecasting.models.temporal_fusion_transformer._tft_forecaster_v2 import (
        TFTForecaster,
    )

    forecaster = TFTForecaster(
        hidden_size=64,
        num_layers=2,
        attention_head_size=4,
        loss=MAE(),
        logging_metrics=[SMAPE()],
        optimizer="adam",
        optimizer_params={"lr": 1e-3},
        trainer=Trainer(max_epochs=5, accelerator="auto"),
        datamodule=EncoderDecoderTimeSeriesDataModule(
            max_encoder_length=30,
            max_prediction_length=5,
            target_normalizer="auto",
        ),
    )


Training
--------

``fit`` takes a ``TimeSeries``. It attaches the data to a copy of the datamodule,
splits the series, fits the normalizer and scalers on the training series, builds
the model and trains it.

.. code-block:: python

    best_ckpt = forecaster.fit(dataset)

* ``trainer``: a trainer used as it is for this call, in place of a copy of the one
  passed to the constructor.
* ``save_ckpt``: whether to save the best checkpoint. Defaults to ``True``.
* ``ckpt_dir``: where checkpoints go. Defaults to ``"checkpoints"``.
* ``ckpt_kwargs``: extra arguments for Lightning's ``ModelCheckpoint``. By default
  it keeps the single checkpoint with the lowest ``val_loss``.
* any other keyword argument is passed on to ``trainer.fit``.

``fit`` returns the path of the best checkpoint, or ``None`` when ``save_ckpt`` is
``False``.


Prediction
----------

``predict`` takes a ``TimeSeries`` and forecasts every series in it. The data is
scaled with the transforms fitted during ``fit``.

.. code-block:: python

    preds = forecaster.predict(dataset, return_info=["y"])
    df = preds.to_pandas()

The result is a ``TimeSeries`` with one row per forecast step. Its columns are
``_series``, the position of the series in the input, ``_time_idx``, the time
step, and one column per target. ``preds.get_metadata().is_prediction`` is
``True``.

* ``mode``: ``"prediction"`` for point forecasts, ``"quantiles"`` for quantile
  forecasts, or ``"raw"`` for the model's output dict. Quantile columns are named
  ``y0``, ``y1`` and so on, all quantiles of the first target before those of the
  next.
* ``return_info``: extra model outputs to return, such as ``"x"``, ``"y"`` or
  ``"index"``. ``"y"`` adds one ``<target>_actual`` column per target. The others
  are stored in ``preds.info``.
* ``output_dir``: if given, the result is also pickled to ``predictions.pkl`` in
  that directory.

Predictions are in the scale of the fitted target normalizer. The inverse
transform is not applied yet.


Checkpoints
-----------

When ``save_ckpt`` is on, ``fit`` writes ``model_cfg.pkl``, ``datamodule_cfg.pkl``
and ``metadata.pkl`` next to the checkpoint. Passing ``ckpt_path`` reads them back,
so a loaded forecaster can predict without any other arguments.

.. code-block:: python

    loaded = TFTForecaster(ckpt_path=best_ckpt)
    preds = loaded.predict(dataset)


Writing a forecaster
--------------------

A new model needs a forecaster class next to it. It implements ``get_cls`` to
return the model class, ``get_datamodule_cls`` to return its datamodule class,
and ``get_model_params`` to return the model arguments. The ``_tags`` dictionary
fills the :doc:`model overview <model_list>` table. ``get_test_train_params``
supplies the settings the test suite runs, and its first entry must be ``{}``.
``TFTForecaster`` in ``models/temporal_fusion_transformer/_tft_forecaster_v2.py``
is a complete example.

.. autoclass:: pytorch_forecasting.models.base._base_forecaster.BaseForecaster
   :noindex:
   :members: fit, predict


API Reference
-------------

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api

   models.base._base_forecaster.BaseForecaster
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
