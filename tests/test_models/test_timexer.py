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
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.encoders import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE, MultiLoss, QuantileLoss
from pytorch_forecasting.models import TimeXer


def _integration(dataloader, tmp_path, loss=None, trainer_kwargs=None, **kwargs):
    """
    Integration test for the TimeXer model.
    Args:
        dataloader: The dataloader to use for training and validation.
        tmp_path: The temporary path to save the model.
        loss: The loss function to use. If None, a default loss function is used.
        trainer_kwargs: Additional arguments for the trainer.
        **kwargs: Additional arguments for the TimeXer model.
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

    # n_targets = len(train_dataloader.dataset.target_positions)

    # resolve the loss function if the loss is not provided explicitly
    if loss is not None:
        pass  # do nothing'
    elif isinstance(train_dataloader.dataset.target_normalizer, MultiNormalizer):
        n_targets = len(train_dataloader.dataset.target_normalizer.normalizers)
        loss = MultiLoss([MAE()] * n_targets)
    else:
        loss = MAE()

    net = TimeXer.from_dataset(
        train_dataloader.dataset,
        hidden_size=kwargs.get("hidden_size", 16),
        n_heads=2,
        e_layers=1,
        d_ff=32,
        patch_length=2,
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

        # test the checkpointing feature
        net = TimeXer.load_from_checkpoint(
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

        # raw prediction if debugging the model

        net.predict(
            val_dataloader,
            return_index=True,
            return_x=True,
            fast_dev_run=True,
            mode="raw",
            trainer_kwargs=trainer_kwargs,
        )

    finally:
        # remove the temporary directory created for the test
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_integration(data_with_covariates, tmp_path):
    """
    Test simple integration of the TimeXer model with a dataloader.
    Args:
        tmp_path: The temporary path to save the model.
        dataloaders: The dataloaders to use for training and validation.
    """

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
    """
    Test the TimeXer model with quantile loss.
    Args:
        data_with_covariates: The data to use for training and validation.
        tmp_path: The temporary path to save the model.
    """

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
    """
    Test TimeXer with multiple target variables.
    Args:
        data_with_covariates: The data to use for training and validation.
        tmp_path: The temporary path to save the model.
    """
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
        features="M",
        trainer_kwargs=dict(accelerator="cpu"),
    )


@pytest.fixture
def model(dataloaders_with_covariates):
    """Create a TimeXer model for testing."""

    dataset = dataloaders_with_covariates["train"].dataset
    net = TimeXer.from_dataset(
        dataset,
        learning_rate=0.01,
        hidden_size=16,
        n_heads=2,
        e_layers=1,
        d_ff=32,
        patch_length=2,
        dropout=0.1,
        loss=MAE(),
    )
    return net


def test_model_init(dataloaders_with_covariates):
    """Test model intialization from a dataset with different params."""
    dataset = dataloaders_with_covariates["train"].dataset

    context_length = dataset.max_encoder_length
    # obtains the patch length from the context length, to ensure that the
    # model can handle large patch lengths
    patch_length_from_context = min(context_length, 2)

    model1 = TimeXer.from_dataset(dataset, patch_length=patch_length_from_context)
    assert isinstance(model1, TimeXer)

    model2 = TimeXer.from_dataset(
        dataset,
        hidden_size=32,
        n_heads=4,
        e_layers=2,
        d_ff=64,
        patch_length=2,
        dropout=0.2,
    )
    # Testing correctness of core params
    assert isinstance(model2, TimeXer)
    assert model2.hparams.hidden_size == 32
    assert model2.hparams.n_heads == 4
    assert model2.hparams.e_layers == 2
    assert model2.hparams.d_ff == 64
    assert model2.hparams.patch_length == 2


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
    """Test prediction with dataloader and various options."""
    val_dataloader = dataloaders_with_covariates["val"]
    model.predict(val_dataloader, fast_dev_run=True, **kwargs)


def test_prediction_with_dataset(model, dataloaders_with_covariates):
    """Test prediction with dataset directly."""
    val_dataloader = dataloaders_with_covariates["val"]
    model.predict(val_dataloader.dataset, fast_dev_run=True)


def test_prediction_with_dataframe(model, data_with_covariates):
    """Test prediction with dataframe directly."""
    model.predict(data_with_covariates, fast_dev_run=True)


def check_embedding_shapes(model):
    """Test that embedding components are initialized correctly."""
    # Check en_embedding
    assert hasattr(model, "en_embedding")
    assert model.en_embedding.hidden_size == model.hparams.hidden_size
    assert model.en_embedding.patch_len == model.hparams.patch_length

    assert hasattr(model, "ex_embedding")
    assert model.ex_embedding.hidden_size == model.hparams.hidden_size
    assert model.ex_embedding.embed_type == model.hparams.embed_type

    assert hasattr(model, "encoder")
    assert len(model.encoder.encoders) == model.hparams.e_layers

    assert hasattr(model, "head")
    assert model.head.n_targets == model.enc_in
    assert model.head.pred_len == model.hparams.prediction_length


def test_no_exogenous_variables():
    """Test model with no exogenous variables."""
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

    forecaster = TimeXer.from_dataset(
        training_dataset,
        hidden_size=16,
        n_heads=2,
        e_layers=1,
        patch_length=2,
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

    # Make predictions
    predictions = forecaster.predict(
        validation_data_loader,
        return_x=True,
        return_y=True,
    )

    assert isinstance(predictions.output, torch.Tensor)
    assert predictions.output.ndim == 2


def test_with_exogenous_variables(tmp_path):
    data = pd.DataFrame(
        {
            "target": np.sin(np.arange(500)) + np.random.normal(0, 0.1, 500),
            "exog": np.cos(np.arange(500)),
            "group_id": np.repeat(np.arange(5), 100),
            "time_idx": np.tile(np.arange(100), 5),
        }
    )

    max_encoder_length = 20
    max_prediction_length = 10
    training_cutoff = 80

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        min_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        time_varying_known_reals=["exog"],
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["group_id"]),
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, data, min_prediction_idx=training_cutoff + 1, stop_randomization=True
    )

    batch_size = 5  # Exactly matches the number of groups
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0, shuffle=False
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, shuffle=False
    )

    model = TimeXer.from_dataset(
        training,
        hidden_size=16,
        n_heads=2,
        e_layers=1,
        patch_length=5,
        dropout=0.1,
    )

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="cpu",
    )

    try:
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # Test direct model forward pass
        batch = next(iter(val_dataloader))
        x, y = batch

        # The purpose of the below code is to ensure that the model
        # treats exogenous variables correctly. The approach here uses
        # masking to nullify exogenous and to confirm that the output is
        # different from the original output (with exogenous variables).
        with torch.no_grad():
            normal_output = model(x)
            x_no_exog = x.copy()
            target_pos = model.target_positions[0]
            mask = torch.ones(
                x_no_exog["encoder_cont"].shape[-1],
                dtype=torch.bool,
                device=x_no_exog["encoder_cont"].device,
            )
            mask[target_pos] = False

            for i in range(x_no_exog["encoder_cont"].shape[-1]):
                if i != target_pos:
                    x_no_exog["encoder_cont"][:, :, i] = 0.0

            no_exog_output = model(x_no_exog)

        assert not torch.allclose(
            normal_output["prediction"], no_exog_output["prediction"], atol=1e-2
        )

        # Test predict API
        predictions = model.predict(
            val_dataloader,
            return_x=True,
            return_y=True,
        )

        assert isinstance(predictions.output, torch.Tensor)
        assert predictions.output.shape[1] == max_prediction_length

    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
