import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting.models.nbeats._nbeatskan import NBeatsKAN


def test_nbeats_kan_integration(dataloaders_fixed_window_without_covariates, tmp_path):
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

    test_kan_num = 10

    net = NBeatsKAN.from_dataset(
        train_dataloader.dataset,
        learning_rate=0.15,
        log_gradient_flow=True,
        widths=[4, 4, 4],
        log_interval=1000,
        backcast_loss_ratio=1.0,
        num=test_kan_num,
        k=3,
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
        net = NBeatsKAN.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

        # check prediction
        net.predict(
            val_dataloader,
            fast_dev_run=True,
            return_index=True,
            return_decoder_lengths=True,
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
