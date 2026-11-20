Forecaster Layer v2
===================

.. warning::
    Please note that the v2 modules are currently in active development and are in beta.
    Please use this API with caution.
    See the complete v2 API documentation :doc:`here <api_v2>` and the stable v1 documentation :doc:`here <api>`.

The v2 Forecaster layer provides a unified interface for training models and generating
predictions from :class:`~pytorch_forecasting.data.TimeSeries` objects. Each forecaster
combines a forecasting model with its training configuration, allowing users to follow
a consistent workflow across different model architectures.

BaseForecaster
--------------

:class:`~pytorch_forecasting.models.base.BaseForecaster` is the base class for v2
forecasters. It provides a common interface for fitting models and generating
predictions.

A forecaster can be configured with a Lightning trainer and a model-specific data
module. The data module defines settings such as encoder length, prediction length,
batch size, and data splitting.

The main methods are:

- ``fit(data)``: Trains the model using a ``TimeSeries`` object. By default, the
  best checkpoint is saved, and its path is returned. Set ``save_ckpt=False``
  to disable checkpoint saving.
- ``predict(data)``: Generates predictions for a ``TimeSeries`` object. The
  default prediction mode returns a ``TimeSeries`` object. The ``quantiles``
  mode also returns a ``TimeSeries``, while the ``raw`` mode returns a dictionary
  of tensors.

For prediction and quantile modes, the returned ``TimeSeries`` contains one row
per forecast step. Additional outputs can be requested through ``return_info``.
These are stored in the ``info`` dictionary, except for ``"y"``, which is added
to the data as actual target columns.

The returned predictions are in the space of the fitted target normalizer.
To obtain a pandas DataFrame, call ``to_pandas()`` on the returned
``TimeSeries`` object.

.. autoclass:: pytorch_forecasting.models.base.BaseForecaster
   :members: __init__, fit, predict

Vignette
--------

The following example demonstrates how to prepare a ``TimeSeries``, configure
a :class:`~pytorch_forecasting.models.temporal_fusion_transformer.TFTForecaster`,
train it, and generate predictions.

First, prepare a simple time series dataset:

.. code-block:: python

    import numpy as np
    import pandas as pd

    from pytorch_forecasting.data import TimeSeries

    n_series = 10
    n_timesteps = 60

    time_idx = np.tile(np.arange(n_timesteps), n_series)
    series = np.repeat(
        [f"series_{i}" for i in range(n_series)],
        n_timesteps,
    )

    target = (
        np.sin(time_idx * 0.1)
        + np.repeat(np.arange(n_series) * 0.1, n_timesteps)
        + time_idx * 0.02
    )

    data = pd.DataFrame(
        {
            "series": series,
            "time_idx": time_idx,
            "target": target,
        }
    )

    time_series = TimeSeries(
        data,
        time="time_idx",
        target="target",
        group=["series"],
    )

Next, configure the data module and forecaster:

.. code-block:: python

    from lightning import Trainer

    from pytorch_forecasting.data.data_module import (
        EncoderDecoderTimeSeriesDataModule,
    )
    from pytorch_forecasting.models.temporal_fusion_transformer import (
        TFTForecaster,
    )

    datamodule = EncoderDecoderTimeSeriesDataModule(
        max_encoder_length=12,
        max_prediction_length=3,
        batch_size=8,
        train_val_test_split=(0.7, 0.15, 0.15),
    )

    forecaster = TFTForecaster(
        hidden_size=16,
        attention_head_size=2,
        trainer=Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        ),
        datamodule=datamodule,
    )

Train the forecaster and generate predictions:

.. code-block:: python

    forecaster.fit(time_series, save_ckpt=False)

    predictions = forecaster.predict(time_series)

The default prediction result is a ``TimeSeries`` object. Convert it to a
pandas DataFrame when needed:

.. code-block:: python

    predictions_df = predictions.to_pandas()

For model outputs that are needed in their raw form, use ``mode="raw"``:

.. code-block:: python

    raw_predictions = forecaster.predict(time_series, mode="raw")

Forecaster Implementations
--------------------------

The following forecasters are available in v2:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   pytorch_forecasting.models.dlinear.DLinearForecaster
   pytorch_forecasting.models.frets.FreTSForecaster
   pytorch_forecasting.models.mlp.DecoderMLPForecaster
   pytorch_forecasting.models.modern_tcn.ModernTCNForecaster
   pytorch_forecasting.models.nhits.NHiTSForecaster
   pytorch_forecasting.models.patch_tst.PatchTSTForecaster
   pytorch_forecasting.models.samformer.SamformerForecaster
   pytorch_forecasting.models.scinet.SCINetForecaster
   pytorch_forecasting.models.softs.SOFTSForecaster
   pytorch_forecasting.models.temporal_fusion_transformer.TFTForecaster
   pytorch_forecasting.models.tide._tide_dsipts.TIDEForecaster
   pytorch_forecasting.models.timexer.TimeXerForecaster
   pytorch_forecasting.models.tsmixer.TSMixerForecaster
   pytorch_forecasting.models.units.UniTSForecaster
   pytorch_forecasting.models.xlstm._xlstm_forecaster_v2.xLSTMTimeForecaster
