import pickle
import shutil

import pytest
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from test_models.conftest import make_dataloaders
from torch.optim import SGD
from torchmetrics import MeanSquaredError

from pytorch_forecasting.metrics import MAE, CrossEntropy, MultiLoss, QuantileLoss
from pytorch_forecasting.models import DecoderMLP


def _integration(data_with_covariates, tmp_path, gpus, data_loader_kwargs={}, train_only=False, **kwargs):
    data_loader_default_kwargs = dict(
        target="target",
        time_varying_known_reals=["price_actual"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["agency"],
        add_relative_time_idx=True,
    )
    data_loader_default_kwargs.update(data_loader_kwargs)
    dataloaders_with_covariates = make_dataloaders(data_with_covariates, **data_loader_default_kwargs)
    train_dataloader = dataloaders_with_covariates["train"]
    val_dataloader = dataloaders_with_covariates["val"]
    test_dataloader = dataloaders_with_covariates["test"]
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min", strict=False
    )

    logger = TensorBoardLogger(tmp_path)
    trainer = pl.Trainer(
        max_epochs=3,
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

    net = DecoderMLP.from_dataset(
        train_dataloader.dataset,
        learning_rate=0.015,
        log_gradient_flow=True,
        log_interval=1000,
        hidden_size=10,
        **kwargs
    )
    net.size()
    try:
        if train_only:
            trainer.fit(net, train_dataloaders=train_dataloader)
        else:
            trainer.fit(
                net,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
        # check loading
        net = DecoderMLP.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        # check prediction
        net.predict(val_dataloader, fast_dev_run=True, return_index=True, return_decoder_lengths=True)
        # check test dataloader
        test_outputs = trainer.test(net, dataloaders=test_dataloader)
        assert len(test_outputs) > 0
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    net.predict(val_dataloader, fast_dev_run=True, return_index=True, return_decoder_lengths=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(train_only=True),
        dict(
            loss=MultiLoss([QuantileLoss(), MAE()]),
            data_loader_kwargs=dict(
                time_varying_unknown_reals=["volume", "discount"],
                target=["volume", "discount"],
            ),
        ),
        dict(
            loss=CrossEntropy(),
            data_loader_kwargs=dict(
                target="agency",
            ),
        ),
        dict(optimizer="SGD", weight_decay=1e-3),
        dict(optimizer=lambda params, lr: SGD(params, lr=lr, weight_decay=1e-3)),
        dict(loss=MeanSquaredError()),
    ],
)
def test_integration(data_with_covariates, tmp_path, gpus, kwargs):
    _integration(data_with_covariates.assign(target=lambda x: x.volume), tmp_path, gpus, **kwargs)


@pytest.fixture
def model(dataloaders_with_covariates):
    dataset = dataloaders_with_covariates["train"].dataset
    net = DecoderMLP.from_dataset(
        dataset,
        learning_rate=0.15,
        log_gradient_flow=True,
        log_interval=1000,
        hidden_size=10,
    )
    return net


def test_pickle(model):
    pkl = pickle.dumps(model)
    pickle.loads(pkl)
