import pickle
import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
import torch

from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.models import TiDEModel
from pytorch_forecasting.tests._conftest import make_dataloaders


def _integration(
    estimator_cls,
    data_with_covariates,
    tmp_path,
    data_loader_kwargs={},
    clip_target: bool = False,
    trainer_kwargs=None,
    **kwargs,
):
    data_with_covariates = data_with_covariates.copy()
    if clip_target:
        data_with_covariates["target"] = data_with_covariates["volume"].clip(1e-3, 1.0)
    else:
        data_with_covariates["target"] = data_with_covariates["volume"]
    data_loader_default_kwargs = dict(
        target="target",
        time_varying_known_reals=["price_actual"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["agency"],
        add_relative_time_idx=True,
    )
    data_loader_default_kwargs.update(data_loader_kwargs)
    dataloaders_with_covariates = make_dataloaders(
        data_with_covariates, **data_loader_default_kwargs
    )

    train_dataloader = dataloaders_with_covariates["train"]
    val_dataloader = dataloaders_with_covariates["val"]
    test_dataloader = dataloaders_with_covariates["test"]

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min"
    )

    logger = TensorBoardLogger(tmp_path)
    if trainer_kwargs is None:
        trainer_kwargs = {}
    trainer = pl.Trainer(
        max_epochs=3,
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

    net = estimator_cls.from_dataset(
        train_dataloader.dataset,
        hidden_size=5,
        learning_rate=0.01,
        log_gradient_flow=True,
        log_interval=1000,
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
        net = estimator_cls.load_from_checkpoint(
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

    net.predict(
        val_dataloader,
        fast_dev_run=True,
        return_index=True,
        return_decoder_lengths=True,
        trainer_kwargs=trainer_kwargs,
    )


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
    not _check_soft_dependencies("matplotlib", severity="none"),
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


def _past_covariates_dataset(**kwargs):
    rng = np.random.default_rng(0)
    n_groups, n_steps = 4, 40
    data = pd.DataFrame(
        {
            "group_id": np.repeat(np.arange(n_groups), n_steps),
            "time_idx": np.tile(np.arange(n_steps), n_groups),
            "target": rng.normal(size=n_groups * n_steps),
            "past_1": rng.normal(size=n_groups * n_steps),
            "past_2": rng.normal(size=n_groups * n_steps),
            "known_1": rng.normal(size=n_groups * n_steps),
            "known_2": rng.normal(size=n_groups * n_steps),
            "past_cat": rng.choice(["a", "b"], size=n_groups * n_steps),
            "known_cat": rng.choice(["c", "d", "e"], size=n_groups * n_steps),
        }
    )
    return TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=10,
        max_prediction_length=5,
        **kwargs,
    )


@pytest.mark.parametrize(
    "dataset_kwargs",
    [
        dict(
            time_varying_unknown_reals=["target", "past_1"],
            time_varying_known_reals=["known_1"],
        ),
        dict(time_varying_unknown_reals=["target", "past_1", "past_2"]),
        dict(
            time_varying_unknown_reals=["target", "past_1"],
            time_varying_known_reals=["known_1"],
            time_varying_unknown_categoricals=["past_cat"],
            time_varying_known_categoricals=["known_cat"],
        ),
    ],
)
def test_past_covariates(dataset_kwargs):
    # covariates known only in the past, i.e. unknown reals or categoricals
    # besides the target, failed with a tensor size mismatch (#1842)
    dataset = _past_covariates_dataset(**dataset_kwargs)
    model = TiDEModel.from_dataset(dataset, hidden_size=8)
    prediction = model.predict(
        dataset.to_dataloader(train=False, batch_size=8),
        trainer_kwargs=dict(logger=False, enable_progress_bar=False),
    )
    assert prediction.shape == (len(dataset), 5)
    assert torch.isfinite(prediction).all()


def test_past_covariates_input_layout():
    # the module reads the target, then the past covariates, then the history
    # of the known covariates, which has to line up with their future values
    dataset = _past_covariates_dataset(
        time_varying_unknown_reals=["past_1", "target", "past_2"],
        time_varying_known_reals=["known_1", "known_2"],
    )
    model = TiDEModel.from_dataset(dataset, hidden_size=8)
    assert model.model.past_cov_dim == 2
    assert model.model.future_cov_dim == 2

    module_inputs = []
    model.model.register_forward_pre_hook(
        lambda module, args: module_inputs.append(args[0])
    )
    x, _ = next(iter(dataset.to_dataloader(train=False, batch_size=4)))
    model(x)

    x_past, x_future, _ = module_inputs[0]

    def columns(cont, names):
        return cont[..., [dataset.reals.index(name) for name in names]]

    expected_past = columns(
        x["encoder_cont"], ["target", "past_1", "past_2", "known_1", "known_2"]
    )
    torch.testing.assert_close(x_past, expected_past)
    torch.testing.assert_close(
        x_future, columns(x["decoder_cont"], ["known_1", "known_2"])
    )
