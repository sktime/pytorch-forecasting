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
from pytorch_forecasting.models import NHiTS
from pytorch_forecasting.utils._dependencies import _get_installed_packages


def _integration(dataloader, tmp_path, trainer_kwargs=None, **kwargs):
    train_dataloader = dataloader["train"]
    val_dataloader = dataloader["val"]
    test_dataloader = dataloader["test"]

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min"
    )

    logger = TensorBoardLogger(tmp_path)
    if trainer_kwargs is None:
        trainer_kwargs = {}
    trainer = pl.Trainer(
        max_epochs=2,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        enable_checkpointing=True,
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        logger=logger,
        **trainer_kwargs,
    )

    kwargs.setdefault("learning_rate", 0.15)
    kwargs.setdefault("weight_decay", 1e-2)

    net = NHiTS.from_dataset(
        train_dataloader.dataset,
        log_gradient_flow=True,
        log_interval=1000,
        hidden_size=8,
        **kwargs,
    )
    net.size()
    try:
        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        # todo: testing somehow disables grad computation even though
        # it is explicitly turned on
        #       loss is calculated as "grad" for MQF2
        if not isinstance(net.loss, MQF2DistributionLoss):
            test_outputs = trainer.test(net, dataloaders=test_dataloader)
            assert len(test_outputs) > 0
        # check loading
        net = NHiTS.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        # check prediction
        net.predict(
            val_dataloader,
            fast_dev_run=True,
            return_index=True,
            return_decoder_lengths=True,
            trainer_kwargs=trainer_kwargs,
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    net.predict(
        val_dataloader,
        fast_dev_run=True,
        return_index=True,
        return_decoder_lengths=True,
    )


LOADERS = [
    "with_covariates",
    "different_encoder_decoder_size",
    "fixed_window_without_covariates",
    "multi_target",
    "quantiles",
    "implicit-quantiles",
]

if "cpflows" in _get_installed_packages():
    LOADERS += ["multivariate-quantiles"]


@pytest.mark.parametrize("dataloader", LOADERS)
def test_integration(
    dataloaders_with_covariates,
    dataloaders_with_different_encoder_decoder_length,
    dataloaders_fixed_window_without_covariates,
    dataloaders_multi_target,
    tmp_path,
    dataloader,
):
    kwargs = {}
    if dataloader == "with_covariates":
        dataloader = dataloaders_with_covariates
        kwargs["backcast_loss_ratio"] = 0.5
    elif dataloader == "different_encoder_decoder_size":
        dataloader = dataloaders_with_different_encoder_decoder_length
    elif dataloader == "fixed_window_without_covariates":
        dataloader = dataloaders_fixed_window_without_covariates
    elif dataloader == "multi_target":
        dataloader = dataloaders_multi_target
        kwargs["loss"] = QuantileLoss()
    elif dataloader == "quantiles":
        dataloader = dataloaders_with_covariates
        kwargs["loss"] = QuantileLoss()
    elif dataloader == "implicit-quantiles":
        dataloader = dataloaders_with_covariates
        kwargs["loss"] = ImplicitQuantileNetworkDistributionLoss()
    elif dataloader == "multivariate-quantiles":
        dataloader = dataloaders_with_covariates
        kwargs["loss"] = MQF2DistributionLoss(
            prediction_length=dataloader["train"].dataset.max_prediction_length
        )
        kwargs["learning_rate"] = 1e-9
        kwargs["trainer_kwargs"] = dict(accelerator="cpu")
    else:
        raise ValueError(f"dataloader {dataloader} unknown")
    _integration(dataloader, tmp_path=tmp_path, **kwargs)


@pytest.fixture(scope="session")
def model(dataloaders_with_covariates):
    dataset = dataloaders_with_covariates["train"].dataset
    net = NHiTS.from_dataset(
        dataset,
        learning_rate=0.15,
        hidden_size=8,
        log_gradient_flow=True,
        log_interval=1000,
        backcast_loss_ratio=1.0,
    )
    return net


def test_pickle(model):
    pkl = pickle.dumps(model)
    pickle.loads(pkl)  # noqa : S301


@pytest.mark.skipif(
    "matplotlib" not in _get_installed_packages(),
    reason="skip test if required package matplotlib not installed",
)
def test_interpretation(model, dataloaders_with_covariates):
    raw_predictions = model.predict(
        dataloaders_with_covariates["val"], mode="raw", return_x=True, fast_dev_run=True
    )
    model.plot_prediction(
        raw_predictions.x, raw_predictions.output, idx=0, add_loss_to_title=True
    )
    model.plot_interpretation(raw_predictions.x, raw_predictions.output, idx=0)


# Bug when max_prediction_length=1 #1571
@pytest.mark.parametrize("max_prediction_length", [1, 5])
def test_prediction_length(max_prediction_length: int):
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
    forecaster = NHiTS.from_dataset(training_dataset, log_val_interval=1)
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
