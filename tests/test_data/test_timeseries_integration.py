import pickle
import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import pytest

from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.metrics import MQF2DistributionLoss, QuantileLoss
from pytorch_forecasting.metrics.distributions import (
    ImplicitQuantileNetworkDistributionLoss,
)
from pytorch_forecasting.models import (
    Baseline, 
    DeepAR, 
    DecoderMLP,
    NBeats,
    NHiTS,
    GRU,
    LSTM,
    MultiEmbedding,
    get_rnn,
    RecurrentNetwork,
    TemporalFusionTransformer,
    TiDEModel,
    TimeXer,
)

from pytorch_forecasting.utils._dependencies import _get_installed_packages


MODELS = [
    Baseline, 
    DeepAR, 
    DecoderMLP,
    NBeats,
    NHiTS,
    GRU,
    LSTM,
    MultiEmbedding,
    get_rnn,
    RecurrentNetwork,
    TemporalFusionTransformer,
    TiDEModel,
    TimeXer,
]


@pytest.mark.parametrize("model", MODELS)
def test_integration(model):
    n_timeseries = 10
    time_points = 10
    data = pd.DataFrame(
        data={
            "target": np.random.rand(time_points * n_timeseries),
            "time_varying_known_real_1": np.random.rand(time_points * n_timeseries),
            "time_idx": np.tile(np.arange(time_points), n_timeseries),
            "group_id": np.repeat(np.arange(n_timeseries), time_points),
        }
    )
    training_dataset = TimeSeriesDataSet(
        data=data,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        time_varying_unknown_reals=["target"],
        time_varying_known_reals=(["time_varying_known_real_1"]),
        max_prediction_length=max_prediction_length,
        max_encoder_length=3,
    )
    training_data_loader = training_dataset.to_dataloader(train=True)
    forecaster = model.from_dataset(training_dataset, log_val_interval=1)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=3,
        min_epochs=2,
        limit_train_batches=10,
    )
    trainer.fit(
        forecaster,
        train_dataloaders=training_data_loader,
    )
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, data, stop_randomization=True, predict=True
    )
    validation_data_loader = validation_dataset.to_dataloader(train=False)
    forecaster.predict(
        validation_data_loader,
        fast_dev_run=True,
        return_index=True,
        return_decoder_lengths=True,
    )
