import shutil
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer


def test_integration(dataloaders_with_coveratiates, tmp_path):
    train_dataloader = dataloaders_with_coveratiates["train"]
    val_dataloader = dataloaders_with_coveratiates["val"]
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")

    # check training
    logger = TensorBoardLogger(tmp_path)
    trainer = pl.Trainer(
        max_epochs=3,
        gpus=0,
        weights_summary="top",
        gradient_clip_val=0.1,
        early_stop_callback=early_stop_callback,
        fast_dev_run=True,
        logger=logger,
    )

    net = TemporalFusionTransformer.from_dataset(
        train_dataloader.dataset,
        learning_rate=0.15,
        hidden_size=4,
        attention_head_size=1,
        dropout=0.2,
        hidden_continuous_size=2,
        loss=QuantileLoss(),
        log_interval=5,
        log_val_interval=1,
        log_gradient_flow=True,
        monotone_constaints={"discount_in_percent": +1},
    )
    net.size()
    try:
        trainer.fit(
            net, train_dataloader=train_dataloader, val_dataloaders=val_dataloader,
        )

        # check loading
        fname = f"{trainer.checkpoint_callback.dirpath}/epoch=0.ckpt"
        net = TemporalFusionTransformer.load_from_checkpoint(fname)

        # check prediction
        net.predict(val_dataloader, fast_dev_run=True, return_index=True, return_decoder_lengths=True)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
