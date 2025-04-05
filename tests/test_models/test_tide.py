import pickle
import shutil
import sys

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import pytest
from test_models.conftest import make_dataloaders
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import (
    GroupNormalizer,
    MultiNormalizer,
    NaNLabelEncoder,
)
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    SMAPE,
    CrossEntropy,
    MultiLoss,
    PoissonLoss,
    QuantileLoss,
)
from pytorch_forecasting.metrics.distributions import NegativeBinomialDistributionLoss
from pytorch_forecasting.models import TiDEModel
from pytorch_forecasting.utils._dependencies import _get_installed_packages


def _integration(dataloader, tmp_path, loss=None, trainer_kwargs=None, **kwargs):
    "Integration test for TiDEModel functionality."

    train_dataloader = dataloader["train"]
    val_dataloader = dataloader["val"]
    test_dataloader = dataloader["test"]

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=1,
        verbose=False,
        mode="min",
    )

    logger = TensorBoardLogger(tmp_path)

    if trainer_kwargs is None:
        trainer_kwargs = {}

    trainer = pl.Trainer(
        max_epochs=2,
        gradient_clip_val=0.1,
        callbacks=[early_stop],
        enable_checkpointing=True,
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        logger=logger,
        **trainer_kwargs,
    )

    kwargs.setdefault("learning_rate", 0.15)

    if loss is not None:
        pass
    elif isinstance(train_dataloader.dataset.target_normalizer, NaNLabelEncoder):
        loss = CrossEntropy()
    elif isinstance(train_dataloader.dataset.target_normalizer, MultiNormalizer):
        loss = MultiLoss(
            [
                (
                    (
                        CrossEntropy()
                        if isinstance(normalizer, NaNLabelEncoder)
                        else QuantileLoss()
                    ),
                )
                for normalizer in train_dataloader.dataset.target_normalizer.normalizers
            ]
        )
    else:
        loss = QuantileLoss()

    net = TiDEModel.from_dataset(
        train_dataloader.dataset,
        hidden_size=4,
        decoder_output_dim=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.2,
        loss=loss,
        add_relative_time_idx=False,
        temporal_decoder_hidden=4,
        temporal_width_future=2,
        temporal_hidden_size_future=4,
        log_interval=5,
        log_val_interval=1,
        **kwargs,
    )

    net.size()

    try:
        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        test_outputs = trainer.test(
            net,
            test_dataloaders=test_dataloader,
        )
        assert len(test_outputs) > 0

        net = TiDEModel.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

        predictions = net.predict(
            val_dataloader,
            return_index=True,
            return_x=True,
            return_y=True,
            fast_dev_run=True,
            trainer_kwargs=trainer_kwargs,
        )

        pred_len = len(predictions.index)

        def check(x):
            if isinstance(x, (tuple, list)):
                for xi in x:
                    check(xi)
            elif isinstance(x, dict):
                for xi in x.values():
                    check(xi)
            else:
                assert (
                    pred_len == x.shape[0]
                ), "first dimension should be prediction length"

        check(predictions.output)
        if isinstance(predictions.output, torch.Tensor):
            assert (
                predictions.output.ndim == 2
            ), "shape of predictions should be batch_size x timesteps"
        else:
            assert all(
                p.ndim == 2 for p in predictions.output
            ), "shape of predictions should be batch_size x timesteps"

        check(predictions.output)

        if isinstance(predictions.output, torch.Tensor):
            assert (
                predictions.output.ndim == 2
            ), "shape of predictions should be batch_size x timesteps"
        else:
            assert all(
                p.ndim == 2 for p in predictions.output
            ), "shape of predictions should be batch_size x timesteps"
        check(predictions.x)
        check(predictions.index)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_integration(multiple_dataloaders_with_covariates, tmp_path):
    """Test basic integration of model with covariates."""
    _integration(
        multiple_dataloaders_with_covariates,
        tmp_path,
        trainer_kwargs=dict(accelerator="cpu"),
    )


@pytest.fixture
def model(dataloaders_with_covariates):
    """Create a model for testing."""

    dataset = dataloaders_with_covariates["train"].dataset

    net = TiDEModel.from_dataset(
        dataset=dataset,
        learning_rate=0.15,
        hidden_size=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        decoder_output_dim=4,
        dropout=0.2,
        temporal_decoder_hidden=4,
        temporal_width_future=2,
        temporal_hidden_size_future=4,
        loss=PoissonLoss(),
        output_size=1,
        log_interval=5,
        log_val_interval=1,
    )
    return net


def test_tensorboard_graph_log(dataloaders_with_covariates, model, tmp_path):
    """Test if tensorboard graph can be logged."""
    d = next(iter(dataloaders_with_covariates["train"]))
    logger = TensorBoardLogger("test", str(tmp_path), log_graph=True)
    logger.log_graph(model, d[0])


def test_pickle(model):
    """Test that model can be pickled and unpickled."""
    pkl = pickle.dumps(model)
    pickle.loads(pkl)  # noqa: S301


@pytest.mark.parametrize(
    "kwargs", [dict(mode="dataframe"), dict(mode="series"), dict(mode="raw")]
)
def test_predict_dependency(
    model, dataloaders_with_covariates, data_with_covariates, kwargs
):
    """Test if predict_dependency works correctly."""
    train_dataset = dataloaders_with_covariates["train"].dataset
    data_with_covariates = data_with_covariates.copy()
    dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        data_with_covariates[lambda x: x.agency == data_with_covariates.agency.iloc[0]],
        predict=True,
    )
    model.predict_dependency(dataset, variable="discount", values=[0.1, 0.0], **kwargs)
    model.predict_dependency(
        dataset,
        variable="agency",
        values=data_with_covariates.agency.unique()[:2],
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mode="raw"),
        dict(mode="quantiles"),
        dict(return_index=True),
        dict(return_decoder_lengths=True),
        dict(return_x=True),
        dict(return_y=True),
    ],
)
def test_prediction_with_dataloader(model, dataloaders_with_covariates, kwargs):
    """Test prediction with dataloader."""
    val_dataloader = dataloaders_with_covariates["val"]
    model.predict(val_dataloader, fast_dev_run=True, **kwargs)


def test_prediction_with_dataset(model, dataloaders_with_covariates):
    """Test prediction with dataset."""
    val_dataloader = dataloaders_with_covariates["val"]
    model.predict(val_dataloader.dataset, fast_dev_run=True)


def test_prediction_with_dataframe(model, data_with_covariates):
    """Test the prediction with dataframe."""
    model.predict(data_with_covariates, fast_dev_run=True)


def test_no_exogenous_variable():
    """Test whether model works without exogenous variables."""
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
        min_encoder_length=10,
        min_prediction_length=5,
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
        log_interval=1,
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
