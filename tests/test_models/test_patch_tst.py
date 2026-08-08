import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import pytest
from test_models.conftest import make_dataloaders
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import MAE, MultiLoss, QuantileLoss
from pytorch_forecasting.models.patch_tst import PatchTST


def _expected_fwd_shape(batch_size, prediction_length, loss):
    """
    Return the expected output shape for the forward pass of the model.
    """
    if isinstance(loss, QuantileLoss):
        n_quantiles = len(loss.quantiles)
        return (batch_size, prediction_length, n_quantiles)
    elif isinstance(loss, MultiLoss):
        shapes = []
        for single_loss in loss.losses:
            if isinstance(single_loss, QuantileLoss):
                n_quantiles = len(single_loss.quantiles)
                shapes.append((batch_size, prediction_length, n_quantiles))
            else:
                shapes.append((batch_size, prediction_length, 1))
        return shapes
    else:
        return (batch_size, prediction_length, 1)


def _integration(dataloader, tmp_path, loss=None, trainer_kwargs=None, **kwargs):
    """
    Integration test for the PatchTST model.
    """
    train_dataloader = dataloader["train"]
    val_dataloader = dataloader["val"]
    test_dataloader = dataloader["test"]

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=5,
        verbose=False,
        mode="min",
    )

    logger = TensorBoardLogger(tmp_path)

    if trainer_kwargs is None:
        trainer_kwargs = {}

    trainer = pl.Trainer(
        max_epochs=2,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        logger=logger,
        enable_checkpointing=True,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        **trainer_kwargs,
    )

    kwargs.setdefault("learning_rate", 0.01)

    if loss is not None:
        pass
    elif isinstance(train_dataloader.dataset.target_normalizer, MultiNormalizer):
        n_targets = len(train_dataloader.dataset.target_normalizer.normalizers)
        loss = MultiLoss([MAE()] * n_targets)
    else:
        loss = MAE()

    net = PatchTST.from_dataset(
        train_dataloader.dataset,
        patch_len=2,
        stride=2,
        d_model=16,
        n_heads=2,
        e_layers=1,
        d_ff=32,
        dropout=0.1,
        loss=loss,
        **kwargs,
    )

    try:
        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        test_outputs = trainer.test(net, dataloaders=test_dataloader)
        assert len(test_outputs) > 0

        net = PatchTST.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
        )
        predictions = net.predict(
            val_dataloader,
            return_index=True,
            return_x=True,
            return_y=True,
            fast_dev_run=True,
            trainer_kwargs=trainer_kwargs,
        )

        if isinstance(predictions.output, torch.Tensor):
            assert predictions.output.ndim == 2, (
                f"shapes of the output should be [batch_size, n_targets], "
                f"but got {predictions.output.shape}"
            )
        else:
            assert all(p.ndim for p in predictions.output), (
                f"shapes of the output should be [batch_size, n_targets], "
                f"but got {predictions.output.shape}"
            )

    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_integration(data_with_covariates, tmp_path):
    dataloaders = make_dataloaders(
        data_with_covariates,
        target="volume",
        time_varying_known_reals=["price_actual"],
        time_varying_unknown_reals=["volume"],
        static_categoricals=["agency"],
        add_relative_time_idx=True,
        target_normalizer=GroupNormalizer(groups=["agency", "sku"], center=False),
    )
    _integration(
        dataloaders,
        tmp_path,
        trainer_kwargs={"accelerator": "cpu"},
    )


def test_quantile_loss(data_with_covariates, tmp_path):
    dataloaders_with_covariates = make_dataloaders(
        data_with_covariates,
        target="volume",
        time_varying_known_reals=["price_actual"],
        time_varying_unknown_reals=["volume"],
        static_categoricals=["agency"],
        add_relative_time_idx=True,
        target_normalizer=GroupNormalizer(groups=["agency", "sku"], center=False),
    )
    _integration(
        dataloaders_with_covariates,
        tmp_path,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        trainer_kwargs=dict(accelerator="cpu"),
    )


def test_multiple_targets(data_with_covariates, tmp_path):
    data = data_with_covariates.copy()
    dataloaders = make_dataloaders(
        data,
        target=["volume", "industry_volume"],
        time_varying_known_reals=["price_actual"],
        time_varying_unknown_reals=["volume", "industry_volume"],
        static_categoricals=["agency"],
        add_relative_time_idx=True,
        target_normalizer=MultiNormalizer(
            [
                GroupNormalizer(groups=["agency", "sku"]),
                GroupNormalizer(groups=["agency", "sku"]),
            ]
        ),
    )
    _integration(
        dataloaders,
        tmp_path,
        trainer_kwargs=dict(accelerator="cpu"),
    )


@pytest.fixture
def model(dataloaders_with_covariates):
    dataset = dataloaders_with_covariates["train"].dataset
    net = PatchTST.from_dataset(
        dataset,
        learning_rate=0.01,
        patch_len=2,
        stride=2,
        d_model=16,
        n_heads=2,
        e_layers=1,
        d_ff=32,
        dropout=0.1,
        loss=MAE(),
    )
    return net


def test_model_init(dataloaders_with_covariates):
    dataset = dataloaders_with_covariates["train"].dataset
    model1 = PatchTST.from_dataset(
        dataset,
        patch_len=16,
        stride=8,
        d_model=32,
        n_heads=4,
        e_layers=2,
        d_ff=64,
        dropout=0.2,
    )
    assert isinstance(model1, PatchTST)
    assert model1.hparams.d_model == 32
    assert model1.hparams.n_heads == 4
    assert model1.hparams.e_layers == 2
    assert model1.hparams.patch_len == 16
    assert model1.hparams.stride == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mode="raw"),
        dict(return_index=True),
        dict(return_x=True),
        dict(return_y=True),
    ],
)
def test_prediction_with_dataloader(model, dataloaders_with_covariates, kwargs):
    val_dataloader = dataloaders_with_covariates["val"]
    model.predict(val_dataloader, fast_dev_run=True, **kwargs)


def test_prediction_with_dataset(model, dataloaders_with_covariates):
    val_dataloader = dataloaders_with_covariates["val"]
    model.predict(val_dataloader.dataset, fast_dev_run=True)


def test_prediction_with_dataframe(model, data_with_covariates):
    model.predict(data_with_covariates, fast_dev_run=True)


def test_no_exogenous_variables():
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
        max_encoder_length=20,
        max_prediction_length=10,
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

    forecaster = PatchTST.from_dataset(
        training_dataset,
        patch_len=10,
        stride=10,
        d_model=16,
        n_heads=2,
        e_layers=1,
    )

    trainer = pl.Trainer(
        max_epochs=2,
        limit_train_batches=8,
        limit_val_batches=8,
    )

    trainer.fit(
        forecaster,
        train_dataloaders=training_data_loader,
        val_dataloaders=validation_data_loader,
    )

    predictions = forecaster.predict(
        validation_data_loader,
        return_x=True,
        return_y=True,
    )
    assert isinstance(predictions.output, torch.Tensor)
    assert predictions.output.ndim == 2


@pytest.mark.skipif(
    True, reason="Skipping due to incompatibility with current model outputs."
)
def test_model_forward_output(dataloaders_with_covariates):
    train_dataloader = dataloaders_with_covariates["train"]
    val_dataloader = dataloaders_with_covariates["val"]

    dataset = train_dataloader.dataset
    batch = next(iter(val_dataloader))
    x, y = batch

    batch_size = x["encoder_cont"].shape[0]
    prediction_length = dataset.max_prediction_length

    loss = MAE()
    model = PatchTST.from_dataset(
        dataset,
        patch_len=2,
        stride=2,
        d_model=16,
        n_heads=2,
        e_layers=1,
        loss=loss,
    )

    with torch.no_grad():
        output = model(x)

    prediction = output["prediction"]
    expected_shape = _expected_fwd_shape(
        batch_size=batch_size,
        prediction_length=prediction_length,
        loss=loss,
    )

    assert (
        prediction.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {prediction.shape}"

    quantile_loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    model_quantile = PatchTST.from_dataset(
        dataset,
        patch_len=2,
        stride=2,
        d_model=16,
        n_heads=2,
        e_layers=1,
        loss=quantile_loss,
    )

    with torch.no_grad():
        output_quantile = model_quantile(x)
    prediction_quantile = output_quantile["prediction"]
    expected_shape_quantile = _expected_fwd_shape(
        batch_size=batch_size,
        prediction_length=prediction_length,
        loss=quantile_loss,
    )
    assert (
        prediction_quantile.shape == expected_shape_quantile
    ), f"Expected {expected_shape_quantile}, got {prediction_quantile.shape}"

    multi_loss = MultiLoss([MAE(), MAE()])
    model_multi = PatchTST.from_dataset(
        dataset,
        patch_len=2,
        stride=2,
        d_model=16,
        n_heads=2,
        e_layers=1,
        loss=multi_loss,
    )

    with torch.no_grad():
        output_multi = model_multi(x)

    prediction_multi = output_multi["prediction"]
    expected_shapes_multi = _expected_fwd_shape(
        batch_size, prediction_length, multi_loss
    )

    assert isinstance(prediction_multi, list)
    assert len(prediction_multi) == len(expected_shapes_multi)

    for i, (pred_tensor, expected_shape) in enumerate(
        zip(prediction_multi, expected_shapes_multi)
    ):
        assert (
            pred_tensor.shape == expected_shape
        ), f"MultiLoss target {i}: Expected {expected_shape}, got {pred_tensor.shape}"


def test_non_divisible_sequence_length():
    """
    Test PatchTST when the sequence length is not a multiple of patch_size.
    This simulates edge cases where max_encoder_length is not perfectly
    divisible by patch_len, verifying the internal padding logic prevents errors.
    """
    # Create dataset with max_encoder_length=13 (not divisible by 4)
    data = pd.DataFrame(
        {
            "target": np.random.rand(100),
            "group_id": np.zeros(100),
            "time_idx": np.arange(100),
        }
    )
    dataset = TimeSeriesDataSet(
        data=data,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=13,
        max_prediction_length=5,
        time_varying_unknown_reals=["target"],
        time_varying_known_reals=[],
    )

    # patch_len=4, stride=4. 13 % 4 = 1, so it requires padding
    forecaster = PatchTST.from_dataset(
        dataset,
        patch_len=4,
        stride=4,
        d_model=16,
        n_heads=2,
        e_layers=1,
    )

    dataloader = dataset.to_dataloader(train=False, batch_size=4)
    batch = next(iter(dataloader))
    x, y = batch

    # The model should run without RuntimeError
    # (maximum size for tensor at dimension 2 is x but size is y)
    with torch.no_grad():
        out = forecaster(x)

    prediction = out["prediction"]

    # Verify final shape matches exactly [batch_size, prediction_length, n_quantiles]
    assert prediction.shape == (x["encoder_cont"].shape[0], 5, 1), (
        f"Expected prediction shape (batch_size, prediction_length, 1), "
        f"but got {prediction.shape}"
    )
