Data
====

.. warning::
    The v2 API is in beta and unstable. It may change without warning and is not
    ready for production use.

.. currentmodule:: pytorch_forecasting

:py:class:`~data.timeseries._timeseries_v2.TimeSeries` is the data type of
``pytorch-forecasting``. It holds a pandas DataFrame together with the role of
each column. You pass a ``TimeSeries`` to ``fit`` and ``predict``, and ``predict``
returns one. A datamodule turns a ``TimeSeries`` into model input. It scales the
data, cuts each series into windows, splits the series into train, validation and
test sets and builds the dataloaders.

If you train through a forecaster, you build the ``TimeSeries`` and configure the
datamodule. The forecaster attaches the data and runs the datamodule for you.
See :doc:`forecaster`.


TimeSeries
----------

``TimeSeries`` does no scaling, windowing or batching. It is also a PyTorch
``Dataset``, and each item is one complete series as tensors.

.. code-block:: python

    from pytorch_forecasting.data import TimeSeries

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

The arguments give each column a role.

* ``time``: integer time index. Within a series it should increase by 1 per step.
* ``target``: one column name, or a list of names for several targets.
* ``group``: the columns that identify a series. Leave it out for a single series.
* ``num`` and ``cat``: numerical and categorical columns. Categorical values must
  be integer codes for now, because no encoder is applied yet.
* ``known`` and ``unknown``: time-varying columns whose future values are known,
  or not known, at prediction time.
* ``static``: columns that stay constant within a series.
* ``weight``: an optional column of per-row weights.
* ``data_future``: an optional second DataFrame with future rows. It may only
  hold time, group, weight, known and static columns.

Roles you leave out are inferred. The target is the last column. The time column
is the first column that is not a group, target, static or weight column. Columns
are split into ``num`` and ``cat`` by dtype. A feature not listed in ``known`` is
treated as unknown.

``dataset.get_metadata()`` returns a
:py:class:`~data._metadata.TimeSeriesMetadata` with the column names, types and
known status. The datamodules read it to work out the input shapes.
``dataset.to_pandas()`` returns the data as one DataFrame, with ``data_future``
appended if it was given.


Datamodules
-----------

There are two datamodules. Each model works with one of them, and the
:doc:`model overview <model_list>` shows which.

* :py:class:`~data.data_module._encoder_decoder_data_module.EncoderDecoderTimeSeriesDataModule`:
  encoder-decoder models such as TFT, TiDE and DecoderMLP.
* :py:class:`~data.data_module._tslib_data_module.TslibDataModule`: tslib-style
  models such as DLinear and TimeXer.

A datamodule can be built without data. It then holds only its settings, and
``with_data(time_series)`` returns a copy that holds the data. Forecasters take a
datamodule built this way. To use a datamodule on its own, pass
``time_series=dataset`` to the constructor instead.

.. code-block:: python

    from pytorch_forecasting.data.data_module import (
        EncoderDecoderTimeSeriesDataModule,
    )

    datamodule = EncoderDecoderTimeSeriesDataModule(
        max_encoder_length=30,
        max_prediction_length=5,
        target_normalizer="auto",
        batch_size=32,
    )
    train_dm = datamodule.with_data(dataset)

``train_val_test_split`` assigns whole series at random to the train, validation
and test sets. A series is never cut in time. With the default
``(0.7, 0.15, 0.15)``, 70% of the series go to training. A dataset with one series
uses it for all three sets. With two series, training gets one and validation and
test share the other.

EncoderDecoderTimeSeriesDataModule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each window has ``max_encoder_length`` past steps and ``max_prediction_length``
future steps. A window starts at every time step, and series shorter than one
window are skipped.

The target is scaled only if you ask for it. The default
``target_normalizer=None`` leaves it as it is. With ``"auto"``, ``setup`` picks a
normalizer from the training series. Categorical targets get a ``NaNLabelEncoder``.
Numerical targets get an ``EncoderNormalizer`` when ``max_encoder_length`` is above
20 and the minimum encoder length is above 1, a ``GroupNormalizer`` when the data
has groups, and a ``TorchNormalizer`` otherwise. Strictly positive targets also get
a log transform when their skew is above 2.5, and a relu transform when it is not.

``scalers`` maps continuous feature names to a scaler such as ``StandardScaler`` or
``TorchNormalizer``. Features without an entry are not scaled. Naming a column that
is not a continuous feature raises ``ValueError``.

The normalizer and scalers are fitted on the training series during ``setup``.
Calling ``with_data`` on a fitted datamodule keeps the fitted transforms, so
prediction data is scaled with the training statistics. The new data must have the
same columns and roles, or ``with_data`` raises ``ValueError``.

These parameters are not used for now. Support will be added later.

* ``randomize_length``
* ``allow_missing_timesteps``
* ``add_encoder_length``
* ``categorical_encoders``
* ``add_target_scales``, which only raises a warning for categorical targets

TslibDataModule
~~~~~~~~~~~~~~~

Each window has ``context_length`` past steps and ``prediction_length`` future
steps. ``window_stride`` sets the gap between window starts, and ``shuffle`` turns
shuffling of training batches on or off.

The feature mode is inferred from the data and stored in ``metadata["features"]``.
It is ``"S"`` for one target and no features, ``"MS"`` for one target with
features, and ``"M"`` for several targets. ``freq`` reaches the model through the
metadata.

These parameters are not used for now. Support will be added later. Until then
this datamodule does not scale the data.

* ``target_normalizer``
* ``scalers``
* ``add_target_scales``


API Reference
-------------

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api

   data.timeseries._timeseries_v2.TimeSeries
   data._metadata.TimeSeriesMetadata
   data.data_module._encoder_decoder_data_module.EncoderDecoderTimeSeriesDataModule
   data.data_module._tslib_data_module.TslibDataModule
   data._metadata.EncDecDataModuleMetadata
   data._metadata.TslibDataModuleMetadata
   data.encoders.EncoderNormalizer
   data.encoders.GroupNormalizer
   data.encoders.MultiNormalizer
   data.encoders.NaNLabelEncoder
   data.encoders.TorchNormalizer
   data.samplers.TimeSynchronizedBatchSampler
   data.samplers.GroupedSampler
