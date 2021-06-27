import pickle
import shutil

import pytest
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import MAE, TemporalConvolutionalNetwork


def test_integration(dataloaders_fixed_window_with_covariates, tmp_path, gpus):
    train_dataloader = dataloaders_fixed_window_with_covariates["train"]
    val_dataloader = dataloaders_fixed_window_with_covariates["val"]
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

    net = TemporalConvolutionalNetwork.from_dataset(
        train_dataloader.dataset,
        learning_rate=0.15,
        log_gradient_flow=True,
        conv_dropout=0.0,
        n_hidden_layers=[64, 32, 16],
        kernel_size=3,
        fc_dropout=0.0,
        loss=MAE(),
        log_interval=1,
    )
    net.size()
    try:
        trainer.fit(
            net,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        # check loading
        net = TemporalConvolutionalNetwork.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        # check prediction
        net.predict(val_dataloader, fast_dev_run=True, return_index=True, return_decoder_lengths=True)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    net.predict(val_dataloader, fast_dev_run=True, return_index=True, return_decoder_lengths=True)


@pytest.fixture
def model(dataloaders_fixed_window_with_covariates):
    dataset = dataloaders_fixed_window_with_covariates["train"].dataset
    net = TemporalConvolutionalNetwork.from_dataset(
        dataset,
        learning_rate=0.15,
        log_gradient_flow=True,
        conv_dropout=0.0,
        n_hidden_layers=[64, 32, 16],
        kernel_size=3,
        fc_dropout=0.0,
        loss=MAE(),
        log_interval=1,
    )
    return net


def test_pickle(model):
    pkl = pickle.dumps(model)
    pickle.loads(pkl)
