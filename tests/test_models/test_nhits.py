import pickle
import shutil

import pytest
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting.models import NHiTS


def _integration(dataloader, tmp_path, gpus):
    train_dataloader = dataloader["train"]
    val_dataloader = dataloader["val"]
    test_dataloader = dataloader["test"]

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")

    logger = TensorBoardLogger(tmp_path)
    trainer = pl.Trainer(
        max_epochs=2,
        gpus=gpus,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        enable_checkpointing=True,
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        logger=logger,
    )

    net = NHiTS.from_dataset(
        train_dataloader.dataset,
        learning_rate=0.15,
        log_gradient_flow=True,
        log_interval=1000,
        hidden_size=8,
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
        net = NHiTS.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        # check prediction
        net.predict(val_dataloader, fast_dev_run=True, return_index=True, return_decoder_lengths=True)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    net.predict(val_dataloader, fast_dev_run=True, return_index=True, return_decoder_lengths=True)


@pytest.mark.parametrize("dataloader", ["with_covariates", "fixed_window_without_covariates", "multi-target"])
def test_integration(
    dataloaders_with_covariates,
    dataloaders_fixed_window_without_covariates,
    dataloaders_multi_target,
    tmp_path,
    gpus,
    dataloader,
):
    if dataloader == "with_covariates":
        dataloader = dataloaders_with_covariates
    elif dataloader == "fixed_window_without_covariates":
        dataloader = dataloaders_fixed_window_without_covariates
    elif dataloader == "multi_target":
        dataloader = dataloaders_multi_target
    else:
        raise ValueError(f"dataloader {dataloader} unknown")
    _integration(dataloader, tmp_path=tmp_path, gpus=gpus)


@pytest.fixture(scope="session")
def model(dataloaders_with_covariates):
    dataset = dataloaders_with_covariates["train"].dataset
    net = NHiTS.from_dataset(
        dataset,
        learning_rate=0.15,
        hidden_size=8,
        log_gradient_flow=True,
        log_interval=1000,
        backcast_loss_ratio=1.0,
    )
    return net


def test_pickle(model):
    pkl = pickle.dumps(model)
    pickle.loads(pkl)


def test_interpretation(model, dataloaders_with_covariates):
    raw_predictions, x = model.predict(dataloaders_with_covariates["val"], mode="raw", return_x=True, fast_dev_run=True)
    model.plot_prediction(x, raw_predictions, idx=0, add_loss_to_title=True)
    model.plot_interpretation(x, raw_predictions, idx=0)
