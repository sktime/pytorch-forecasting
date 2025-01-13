import pickle
import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import pytest

from pytorch_forecasting.models import NBeats
from pytorch_forecasting.utils._dependencies import _get_installed_packages


def test_integration(dataloaders_fixed_window_without_covariates, tmp_path):
    train_dataloader = dataloaders_fixed_window_without_covariates["train"]
    val_dataloader = dataloaders_fixed_window_without_covariates["val"]
    test_dataloader = dataloaders_fixed_window_without_covariates["test"]

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min"
    )

    logger = TensorBoardLogger(tmp_path)
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
    )

    net = NBeats.from_dataset(
        train_dataloader.dataset,
        learning_rate=0.15,
        log_gradient_flow=True,
        widths=[4, 4, 4],
        log_interval=1000,
        backcast_loss_ratio=1.0,
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
        net = NBeats.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        # check prediction
        net.predict(
            val_dataloader,
            fast_dev_run=True,
            return_index=True,
            return_decoder_lengths=True,
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    net.predict(
        val_dataloader,
        fast_dev_run=True,
        return_index=True,
        return_decoder_lengths=True,
    )


@pytest.fixture(scope="session")
def model(dataloaders_fixed_window_without_covariates):
    dataset = dataloaders_fixed_window_without_covariates["train"].dataset
    net = NBeats.from_dataset(
        dataset,
        learning_rate=0.15,
        log_gradient_flow=True,
        widths=[4, 4, 4],
        log_interval=1000,
        backcast_loss_ratio=1.0,
    )
    return net


def test_pickle(model):
    pkl = pickle.dumps(model)
    pickle.loads(pkl)  # noqa: S301


@pytest.mark.skipif(
    "matplotlib" not in _get_installed_packages(),
    reason="skip test if required package matplotlib not installed",
)
def test_interpretation(model, dataloaders_fixed_window_without_covariates):
    raw_predictions = model.predict(
        dataloaders_fixed_window_without_covariates["val"],
        mode="raw",
        return_x=True,
        fast_dev_run=True,
    )
    model.plot_interpretation(raw_predictions.x, raw_predictions.output, idx=0)
