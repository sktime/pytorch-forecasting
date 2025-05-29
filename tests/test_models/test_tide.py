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
from pytorch_forecasting.tests.test_all_estimators import _integration
from pytorch_forecasting.utils._dependencies import _get_installed_packages


def _tide_integration(dataloaders, tmp_path, trainer_kwargs=None, **kwargs):
    """TiDE specific wrapper around the common integration test function.

    Args:
        dataloaders: Dictionary of dataloaders for train, val, and test.
        tmp_path: Temporary path for saving the model.
        trainer_kwargs: Additional arguments for the Trainer.
        **kwargs: Additional arguments for the TiDEModel.

    Returns:
        Predictions from the trained model.
    """
    from pytorch_forecasting.tests._data_scenarios import data_with_covariates

    df = data_with_covariates()

    tide_kwargs = {
        "temporal_decoder_hidden": 8,
        "temporal_width_future": 4,
        "dropout": 0.1,
    }

    tide_kwargs.update(kwargs)
    train_dataset = dataloaders["train"].dataset

    data_loader_kwargs = {
        "target": train_dataset.target,
        "group_ids": train_dataset.group_ids,
        "time_varying_known_reals": train_dataset.time_varying_known_reals,
        "time_varying_unknown_reals": train_dataset.time_varying_unknown_reals,
        "static_categoricals": train_dataset.static_categoricals,
        "static_reals": train_dataset.static_reals,
        "add_relative_time_idx": train_dataset.add_relative_time_idx,
    }
    return _integration(
        TiDEModel,
        df,
        tmp_path,
        data_loader_kwargs=data_loader_kwargs,
        trainer_kwargs=trainer_kwargs,
        **tide_kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"loss": SMAPE()},
        {"temporal_decoder_hidden": 16},
        {"dropout": 0.2, "use_layer_norm": True},
    ],
)
def test_integration(dataloaders_with_covariates, tmp_path, kwargs):
    _tide_integration(dataloaders_with_covariates, tmp_path, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
    ],
)
def test_multi_target_integration(dataloaders_multi_target, tmp_path, kwargs):
    _tide_integration(dataloaders_multi_target, tmp_path, **kwargs)


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
        fast_dev_run=True,
        return_x=True,
        return_y=True,
        return_index=True,
    )
