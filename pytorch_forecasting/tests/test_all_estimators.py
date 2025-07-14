"""Automated tests based on the skbase test suite template."""

from inspect import isclass
import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from skbase.testing import BaseFixtureGenerator as _BaseFixtureGenerator

from pytorch_forecasting._registry import all_objects
from pytorch_forecasting.tests._base._fixture_generator import (
    BaseFixtureGenerator,
    PackageConfig,
)
from pytorch_forecasting.tests._config import EXCLUDE_ESTIMATORS, EXCLUDED_TESTS


def _integration(
    estimator_cls,
    dataloaders,
    tmp_path,
    trainer_kwargs=None,
    data_loader_kwargs={},  # only present to capture and pop these from kwargs
    clip_target: bool = False,  # only present to capture and pop these from kwargs
    # todo: refactor this more cleanly to metadata class
    **kwargs,
):
    """Unified integration test for all estimators."""
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]
    test_dataloader = dataloaders["test"]

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

    net = estimator_cls.from_dataset(
        train_dataloader.dataset,
        learning_rate=0.01,
        log_gradient_flow=True,
        log_interval=1000,
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
        net = estimator_cls.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

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


class TestAllPtForecasters(PackageConfig, BaseFixtureGenerator):
    """Generic tests for all objects in the mini package."""

    object_type_filter = "forecaster_pytorch_v1"

    def test_doctest_examples(self, object_class):
        """Runs doctests for estimator class."""
        from skbase.utils.doctest_run import run_doctest

        run_doctest(object_class, name=f"class {object_class.__name__}")

    def test_integration(
        self,
        object_pkg,
        trainer_kwargs,
        tmp_path,
    ):
        """Fails for certain, for testing."""

        object_class = object_pkg.get_model_cls()
        dataloaders = object_pkg._get_test_dataloaders_from(trainer_kwargs)

        _integration(object_class, dataloaders, tmp_path, **trainer_kwargs)

    def test_pkg_linkage(self, object_pkg, object_class):
        """Test that the package is linked correctly."""
        # check name method
        msg = (
            f"Package {object_pkg}.name() does not match class "
            f"name {object_class.__name__}. "
            "The expected package name is "
            f"{object_class.__name__}_pkg."
        )
        assert object_pkg.name() == object_class.__name__, msg

        # check naming convention
        msg = (
            f"Package {object_pkg.__name__} does not match class "
            f"name {object_class.__name__}. "
            "The expected package name is "
            f"{object_class.__name__}_pkg."
        )
        assert object_pkg.__name__ == object_class.__name__ + "_pkg", msg
