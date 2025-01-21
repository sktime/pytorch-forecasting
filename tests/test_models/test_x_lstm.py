import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import pytest

from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.models.x_lstm_time.x_lstm import xLSTMTime


def _integration(
    dataloaders_fixed_window_without_covariates, tmp_path, xlstm_type="slstm", **kwargs
):

    train_dataloader = dataloaders_fixed_window_without_covariates["train"]
    val_dataloader = dataloaders_fixed_window_without_covariates["val"]
    test_dataloader = dataloaders_fixed_window_without_covariates["test"]

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min"
    )

    logger = TensorBoardLogger(tmp_path)
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
    )

    model_kwargs = {
        "input_size": 1,
        "output_size": 1,
        "hidden_size": 32,
        "xlstm_type": xlstm_type,
        "learning_rate": 0.01,
        "loss": SMAPE(),
    }

    model_kwargs.update(kwargs)

    net = xLSTMTime.from_dataset(train_dataloader.dataset, **model_kwargs)

    try:

        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        test_outputs = trainer.test(net, dataloaders=test_dataloader)
        assert len(test_outputs) > 0

        net = xLSTMTime.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

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


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"xlstm_type": "mlstm"},
        {"num_layers": 2},
        {"xlstm_type": "slstm", "input_projection_size": 32},
        {
            "xlstm_type": "mlstm",
            "decomposition_kernel": 13,
            "dropout": 0.2,
        },
    ],
)
def test_integration(dataloaders_fixed_window_without_covariates, tmp_path, kwargs):
    _integration(dataloaders_fixed_window_without_covariates, tmp_path, **kwargs)


@pytest.fixture(scope="session")
def model(dataloaders_fixed_window_without_covariates):
    dataset = dataloaders_fixed_window_without_covariates["train"].dataset
    net = xLSTMTime.from_dataset(
        dataset,
        input_size=1,
        hidden_size=32,
        output_size=1,
        xlstm_type="slstm",
        learning_rate=0.01,
        loss=SMAPE(),
    )
    return net
