import pickle
import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import pytest

from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from pytorch_forecasting.models import TiDEModel
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

    kwargs.setdefault("hidden_size", 16)
    kwargs.setdefault("temporal_decoder_hidden", 8)
    kwargs.setdefault("temporal_width_future", 4)
    kwargs.setdefault("dropout", 0.1)
    kwargs.setdefault("learning_rate", 0.01)

    net = TiDEModel.from_dataset(
        train_dataloader.dataset,
        **kwargs,
    )
    net.size()
    try:
        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        test_outputs = trainer.test(net, dataloaders=test_dataloader)
        assert len(test_outputs) > 0
        # check loading
        net = TiDEModel.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

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

    predictions = net.predict(
        val_dataloader,
        fast_dev_run=True,
        return_index=True,
        return_decoder_lengths=True,
    )
    return predictions


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"loss": SMAPE()},
        {"hidden_size": 32, "temporal_decoder_hidden": 16},
        {"dropout": 0.2, "use_layer_norm": True},
    ],
)
def test_integration(dataloaders_with_covariates, tmp_path, kwargs):
    _integration(dataloaders_with_covariates, tmp_path, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # Default settings for multi-target
    ],
)
def test_multi_target_integration(dataloaders_multi_target, tmp_path, kwargs):
    _integration(dataloaders_multi_target, tmp_path, **kwargs)


@pytest.fixture
def model(dataloaders_with_covariates):
    dataset = dataloaders_with_covariates["train"].dataset
    net = TiDEModel.from_dataset(
        dataset,
        hidden_size=16,
        dropout=0.1,
        temporal_width_future=4,
    )
    return net


def test_pickle(model):
    pkl = pickle.dumps(model)
    pickle.loads(pkl)  # noqa: S301


@pytest.mark.skipif(
    "matplotlib" not in _get_installed_packages(),
    reason="skip test if required package matplotlib not installed",
)
def test_prediction_visualization(model, dataloaders_with_covariates):
    raw_predictions = model.predict(
        dataloaders_with_covariates["val"],
        mode="raw",
        return_x=True,
        fast_dev_run=True,
    )
    model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=0)


def test_prediction_with_kwargs(model, dataloaders_with_covariates):
    # Tests prediction works with different keyword arguments
    model.predict(
        dataloaders_with_covariates["val"], return_index=True, fast_dev_run=True
    )
    model.predict(
        dataloaders_with_covariates["val"],
        return_x=True,
        return_y=True,
        fast_dev_run=True,
    )


def test_no_exogenous_variable():
    data = pd.DataFrame(
        {
            "target": np.ones(1600),
            "group_id": np.repeat(np.arange(16), 100),
            "time_idx": np.tile(np.arange(100), 16),
        }
    )
    training_dataset = TimeSeriesDataSet(
        data=data,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=10,
        max_prediction_length=5,
        time_varying_unknown_reals=["target"],
        time_varying_known_reals=[],
    )
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, data, stop_randomization=True, predict=True
    )
    training_data_loader = training_dataset.to_dataloader(
        train=True, batch_size=8, num_workers=0
    )
    validation_data_loader = validation_dataset.to_dataloader(
        train=False, batch_size=8, num_workers=0
    )
    forecaster = TiDEModel.from_dataset(
        training_dataset,
    )
    from lightning.pytorch import Trainer

    trainer = Trainer(
        max_epochs=2,
        limit_train_batches=8,
        limit_val_batches=8,
    )
    trainer.fit(
        forecaster,
        train_dataloaders=training_data_loader,
        val_dataloaders=validation_data_loader,
    )
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = TiDEModel.load_from_checkpoint(best_model_path)
    best_model.predict(
        validation_data_loader,
        return_x=True,
        return_y=True,
        return_index=True,
    )
