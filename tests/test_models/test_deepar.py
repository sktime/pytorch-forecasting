import pickle
import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import pytest
from test_models.conftest import make_dataloaders
from torch import nn

from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import (
    BetaDistributionLoss,
    ImplicitQuantileNetworkDistributionLoss,
    LogNormalDistributionLoss,
    MultivariateNormalDistributionLoss,
    NegativeBinomialDistributionLoss,
    NormalDistributionLoss,
)
from pytorch_forecasting.models import DeepAR


def _integration(
    data_with_covariates,
    tmp_path,
    cell_type="LSTM",
    data_loader_kwargs={},
    clip_target: bool = False,
    trainer_kwargs=None,
    **kwargs,
):
    data_with_covariates = data_with_covariates.copy()
    if clip_target:
        data_with_covariates["target"] = data_with_covariates["volume"].clip(1e-3, 1.0)
    else:
        data_with_covariates["target"] = data_with_covariates["volume"]
    data_loader_default_kwargs = dict(
        target="target",
        time_varying_known_reals=["price_actual"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["agency"],
        add_relative_time_idx=True,
    )
    data_loader_default_kwargs.update(data_loader_kwargs)
    dataloaders_with_covariates = make_dataloaders(
        data_with_covariates, **data_loader_default_kwargs
    )

    train_dataloader = dataloaders_with_covariates["train"]
    val_dataloader = dataloaders_with_covariates["val"]
    test_dataloader = dataloaders_with_covariates["test"]

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min"
    )

    logger = TensorBoardLogger(tmp_path)
    if trainer_kwargs is None:
        trainer_kwargs = {}
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
        **trainer_kwargs,
    )

    net = DeepAR.from_dataset(
        train_dataloader.dataset,
        hidden_size=5,
        cell_type=cell_type,
        learning_rate=0.01,
        log_gradient_flow=True,
        log_interval=1000,
        n_plotting_samples=100,
        **kwargs,
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
        net = DeepAR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        # check prediction
        net.predict(
            val_dataloader,
            fast_dev_run=True,
            return_index=True,
            return_decoder_lengths=True,
            trainer_kwargs=trainer_kwargs,
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    net.predict(
        val_dataloader,
        fast_dev_run=True,
        return_index=True,
        return_decoder_lengths=True,
        trainer_kwargs=trainer_kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"cell_type": "GRU"},
        dict(
            loss=LogNormalDistributionLoss(),
            clip_target=True,
            data_loader_kwargs=dict(
                target_normalizer=GroupNormalizer(
                    groups=["agency", "sku"], transformation="log"
                )
            ),
        ),
        dict(
            loss=NegativeBinomialDistributionLoss(),
            clip_target=False,
            data_loader_kwargs=dict(
                target_normalizer=GroupNormalizer(
                    groups=["agency", "sku"], center=False
                )
            ),
        ),
        dict(
            loss=BetaDistributionLoss(),
            clip_target=True,
            data_loader_kwargs=dict(
                target_normalizer=GroupNormalizer(
                    groups=["agency", "sku"], transformation="logit"
                )
            ),
        ),
        dict(
            data_loader_kwargs=dict(
                lags={"volume": [2, 5]},
                target="volume",
                time_varying_unknown_reals=["volume"],
                min_encoder_length=2,
            )
        ),
        dict(
            data_loader_kwargs=dict(
                time_varying_unknown_reals=["volume", "discount"],
                target=["volume", "discount"],
                lags={"volume": [2], "discount": [2]},
            )
        ),
        dict(
            loss=ImplicitQuantileNetworkDistributionLoss(hidden_size=8),
        ),
        dict(
            loss=MultivariateNormalDistributionLoss(),
            trainer_kwargs=dict(accelerator="cpu"),
        ),
        dict(
            loss=MultivariateNormalDistributionLoss(),
            data_loader_kwargs=dict(
                target_normalizer=GroupNormalizer(
                    groups=["agency", "sku"], transformation="log1p"
                )
            ),
            trainer_kwargs=dict(accelerator="cpu"),
        ),
    ],
)
def test_integration(data_with_covariates, tmp_path, kwargs):
    if "loss" in kwargs and isinstance(
        kwargs["loss"], NegativeBinomialDistributionLoss
    ):
        data_with_covariates = data_with_covariates.assign(
            volume=lambda x: x.volume.round()
        )
    _integration(data_with_covariates, tmp_path, **kwargs)


@pytest.fixture
def model(dataloaders_with_covariates):
    dataset = dataloaders_with_covariates["train"].dataset
    net = DeepAR.from_dataset(
        dataset,
        hidden_size=5,
        learning_rate=0.15,
        log_gradient_flow=True,
        log_interval=1000,
    )
    return net


def test_predict_average(model, dataloaders_with_covariates):
    prediction = model.predict(
        dataloaders_with_covariates["val"],
        fast_dev_run=True,
        mode="prediction",
        n_samples=100,
    )
    assert prediction.ndim == 2, "expected averaging of samples"


def test_predict_samples(model, dataloaders_with_covariates):
    prediction = model.predict(
        dataloaders_with_covariates["val"],
        fast_dev_run=True,
        mode="samples",
        n_samples=100,
    )
    assert prediction.size()[-1] == 100, "expected raw samples"


@pytest.mark.parametrize(
    "loss", [NormalDistributionLoss(), MultivariateNormalDistributionLoss()]
)
def test_pickle(dataloaders_with_covariates, loss):
    dataset = dataloaders_with_covariates["train"].dataset
    model = DeepAR.from_dataset(
        dataset,
        hidden_size=5,
        learning_rate=0.15,
        log_gradient_flow=True,
        log_interval=1000,
        loss=loss,
    )
    pkl = pickle.dumps(model)
    pickle.loads(pkl)  # noqa: S301
