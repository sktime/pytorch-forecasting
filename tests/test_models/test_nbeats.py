import pickle
import shutil

import pytest
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting.models import NBeats


def test_integration(dataloaders_fixed_window_without_covariates, tmp_path, gpus):
    train_dataloader = dataloaders_fixed_window_without_covariates["train"]
    val_dataloader = dataloaders_fixed_window_without_covariates["val"]
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")

    logger = TensorBoardLogger(tmp_path)
    trainer = pl.Trainer(
        max_epochs=2,
        gpus=gpus,
        weights_summary="top",
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        checkpoint_callback=True,
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
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
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        # check loading
        net = NBeats.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        # check prediction
        net.predict(val_dataloader, fast_dev_run=True, return_index=True, return_decoder_lengths=True)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    net.predict(val_dataloader, fast_dev_run=True, return_index=True, return_decoder_lengths=True)


@pytest.fixture
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
    pickle.loads(pkl)
