import pickle
import shutil

import pytest
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from pytorch_forecasting.models import DeepAR
from pytorch_forecasting.models.deepar.sub_modules import get_cell, TimeSeriesLSTM, TimeSeriesGRU



def _integration(dataloaders_with_covariates, tmp_path, gpus, cell_type="LSTM"):
    train_dataloader = dataloaders_with_covariates["train"]
    val_dataloader = dataloaders_with_covariates["val"]
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")

    logger = TensorBoardLogger(tmp_path)
    checkpoint = ModelCheckpoint(filepath=tmp_path)
    trainer = pl.Trainer(
        checkpoint_callback=checkpoint,
        max_epochs=3,
        gpus=gpus,
        weights_summary="top",
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        fast_dev_run=True,
        logger=logger,
    )

    net = DeepAR.from_dataset(
        train_dataloader.dataset,
        cell_type=cell_type,
        learning_rate=0.15,
        log_gradient_flow=True,
        log_interval=1000,
    )
    net.size()
    try:
        trainer.fit(
            net,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        # check loading
        fname = f"{trainer.checkpoint_callback.dirpath}/epoch=0.ckpt"
        net = DeepAR.load_from_checkpoint(fname)

        # check prediction
        net.predict(val_dataloader, fast_dev_run=True, return_index=True, return_decoder_lengths=True)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    net.predict(val_dataloader, fast_dev_run=True, return_index=True, return_decoder_lengths=True)


def test_integration(dataloaders_with_covariates, tmp_path, gpus):
    _integration(dataloaders_with_covariates, tmp_path, gpus)


def test_integration_with_gru(dataloaders_with_covariates, tmp_path, gpus):
    _integration(dataloaders_with_covariates, tmp_path, gpus, "GRU")


def test_get_lstm_cell():
    cell = get_cell("LSTM")(10, 10)
    assert isinstance(cell, TimeSeriesLSTM)
    assert isinstance(cell, nn.LSTM)


def test_get_gru_cell():
    cell = get_cell("GRU")(10, 10)
    assert isinstance(cell, TimeSeriesGRU)
    assert isinstance(cell, nn.GRU)


def test_get_cell_raises_value_error():
    pytest.raises(ValueError, lambda: get_cell("ABCDEF"))


@pytest.fixture
def model(dataloaders_with_covariates):
    dataset = dataloaders_with_covariates["train"].dataset
    net = DeepAR.from_dataset(
        dataset,
        learning_rate=0.15,
        log_gradient_flow=True,
        log_interval=1000,
    )
    return net


def test_pickle(model):
    pkl = pickle.dumps(model)
    pickle.loads(pkl)
