"""Automated tests based on the skbase test suite template."""

from inspect import isclass
import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import torch.nn as nn

from pytorch_forecasting.tests._conftest import make_dataloaders_v2 as make_dataloaders
from pytorch_forecasting.tests.test_all_estimators import (
    BaseFixtureGenerator,
    PackageConfig,
)

# whether to test only estimators from modules that are changed w.r.t. main
# default is False, can be set to True by pytest --only_changed_modules True flag
ONLY_CHANGED_MODULES = False


def _integration(
    estimator_cls,
    data_with_covariates,
    tmp_path,
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
        group_ids=["agency_encoded", "sku_encoded"],
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
    training_data_module = dataloaders_with_covariates["data_module"]
    metadata = training_data_module.metadata

    assert metadata["encoder_cont"] == 14  # 14 features (8 known + 6 unknown)
    assert metadata["encoder_cat"] == 0
    assert metadata["decoder_cont"] == 8  # 8 (only known features)
    assert metadata["decoder_cat"] == 0
    assert metadata["static_categorical_features"] == 0
    assert (
        metadata["static_continuous_features"] == 2
    )  # 2 (agency_encoded, sku_encoded)
    assert metadata["target"] == 1

    batch_x, batch_y = next(iter(train_dataloader))

    assert batch_x["encoder_cont"].shape[2] == metadata["encoder_cont"]
    assert batch_x["encoder_cat"].shape[2] == metadata["encoder_cat"]

    assert batch_x["decoder_cont"].shape[2] == metadata["decoder_cont"]
    assert batch_x["decoder_cat"].shape[2] == metadata["decoder_cat"]

    if "static_categorical_features" in batch_x:
        assert (
            batch_x["static_categorical_features"].shape[2]
            == metadata["static_categorical_features"]
        )

    if "static_continuous_features" in batch_x:
        assert (
            batch_x["static_continuous_features"].shape[2]
            == metadata["static_continuous_features"]
        )

    assert batch_y.shape[2] == metadata["target"]

    net = estimator_cls(
        metadata=metadata,
        loss=nn.MSELoss(),
        **kwargs,
    )

    try:
        trainer.fit(
            net,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        test_outputs = trainer.test(net, dataloaders=test_dataloader)
        assert len(test_outputs) > 0
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


class TestAllPtForecastersV2(PackageConfig, BaseFixtureGenerator):
    """Generic tests for all objects in the mini package."""

    object_type_filter = "ptf-v2"

    def test_doctest_examples(self, object_class):
        """Runs doctests for estimator class."""
        from skbase.utils.doctest_run import run_doctest

        run_doctest(object_class, name=f"class {object_class.__name__}")

    def test_integration(
        self,
        object_metadata,
        trainer_kwargs,
        tmp_path,
    ):
        from pytorch_forecasting.tests._data_scenarios import data_with_covariates_v2

        data_with_covariates = data_with_covariates_v2()
        object_class = object_metadata.get_model_cls()
        _integration(object_class, data_with_covariates, tmp_path, **trainer_kwargs)
