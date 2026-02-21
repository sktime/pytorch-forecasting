import torch
import pickle
import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import pytest
from test_models.conftest import make_dataloaders
from torchmetrics import MeanSquaredError

from pytorch_forecasting.metrics import (
    MAE,
    BetaDistributionLoss,
    CrossEntropy,
    ImplicitQuantileNetworkDistributionLoss,
    LogNormalDistributionLoss,
    MultiLoss,
    MultivariateNormalDistributionLoss,
    NegativeBinomialDistributionLoss,
    NormalDistributionLoss,
    QuantileLoss,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.models import DecoderMLP


def _integration(
    data_with_covariates, tmp_path, data_loader_kwargs={}, train_only=False, **kwargs
):
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
        monitor="val_loss",
        min_delta=1e-4,
        patience=1,
        verbose=False,
        mode="min",
        strict=False,
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

    net = DecoderMLP.from_dataset(
        train_dataloader.dataset,
        learning_rate=0.015,
        log_gradient_flow=True,
        log_interval=1000,
        hidden_size=10,
        **kwargs,
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
        net = DecoderMLP.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

        # check prediction
        net.predict(
            val_dataloader,
            fast_dev_run=True,
            return_index=True,
            return_decoder_lengths=True,
        )
        # check test dataloader
        test_outputs = trainer.test(net, dataloaders=test_dataloader)
        assert len(test_outputs) > 0
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
        dict(loss=MeanSquaredError()),
        dict(
            loss=MeanSquaredError(),
            data_loader_kwargs=dict(min_prediction_length=1, min_encoder_length=1),
        ),
    ],
)
def test_integration(data_with_covariates, tmp_path, kwargs):
    _integration(
        data_with_covariates.assign(target=lambda x: x.volume), tmp_path, **kwargs
    )


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
    pickle.loads(pkl)  # noqa: S301


@pytest.mark.parametrize(
    "loss_cls, target, data_loader_kwargs, normalizer",
    [
        (NormalDistributionLoss, "volume", {}, None),
        (NegativeBinomialDistributionLoss, "volume", {}, GroupNormalizer(method="identity", center=False)),
        (MultivariateNormalDistributionLoss, "volume", {}, None),
        (LogNormalDistributionLoss, "volume", {}, GroupNormalizer(center=False, transformation="log")),
        (BetaDistributionLoss, "volume", {}, GroupNormalizer(center=True, transformation="logit")),
        (ImplicitQuantileNetworkDistributionLoss, "volume", {}, None),
    ],
)
def test_decoder_mlp_supported_losses(
    data_with_covariates,
    tmp_path,
    loss_cls,
    target,
    data_loader_kwargs,
    normalizer,
):

    data_loader_kwargs = dict(data_loader_kwargs)
    data_loader_kwargs["target"] = target
    if normalizer is not None:
        data_loader_kwargs["target_normalizer"] = normalizer

    # Always copy data to avoid accidental overwrite
    data = data_with_covariates.copy()

    # Preprocess target for distribution requirements
    if loss_cls.__name__ == "NegativeBinomialDistributionLoss":
        # Targets must be strictly non-negative integers (>=0)
        data["target"] = data["target"].apply(lambda x: max(0, int(round(x))))
        # After normalization, cast to int again to ensure integer type
        def to_int_if_possible(x):
            try:
                return int(round(x))
            except Exception:
                return 0
        data["target"] = data["target"].apply(to_int_if_possible)
    elif loss_cls.__name__ == "LogNormalDistributionLoss":
        # Targets must be strictly positive (>0)
        data["target"] = data["target"].apply(lambda x: max(1e-3, float(x)))
        # Ensure no zero or negative values after normalization
        data["target"] = data["target"].apply(lambda x: x if x > 0 else 1e-3)
    elif loss_cls.__name__ == "BetaDistributionLoss":
        # Targets must be strictly between 0 and 1 (not including endpoints)
        data["target"] = data["target"].apply(lambda x: min(max(x, 1e-3), 1-1e-3))

    dataloaders = make_dataloaders(
        data, **data_loader_kwargs
    )
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]

    if loss_cls.__name__ == "NegativeBinomialDistributionLoss":
        original_collate = train_dataloader.collate_fn

        def int_target_collate(batch):
            x, y = original_collate(batch)
            target, weight = y
            if isinstance(target, torch.Tensor):
                y = (target.long(), weight)
            elif isinstance(target, list | tuple):
                y = ([t.long() for t in target], weight)
            return x, y

        train_dataloader.collate_fn = int_target_collate
        val_dataloader.collate_fn = int_target_collate
    net = DecoderMLP.from_dataset(
        train_dataloader.dataset,
        loss=loss_cls(),
        learning_rate=0.01,
        hidden_size=8,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
        limit_train_batches=1,
        limit_val_batches=1,
    )


    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    preds = net.predict(val_dataloader, fast_dev_run=True)
    assert preds is not None
