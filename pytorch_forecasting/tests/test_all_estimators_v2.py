"""Automated tests based on the skbase test suite template."""

import shutil

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.nn as nn

from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.tests.test_all_estimators import (
    EstimatorFixtureGenerator,
    EstimatorPackageConfig,
)

# whether to test only estimators from modules that are changed w.r.t. main
# default is False, can be set to True by pytest --only_changed_modules True flag
ONLY_CHANGED_MODULES = False


def _integration(
    estimator_cls,
    dataloaders,
    tmp_path,
    data_loader_kwargs={},
    clip_target: bool = False,
    trainer_kwargs=None,
    **kwargs,
):
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
    training_data_module = dataloaders.get("data_module")
    metadata = training_data_module.metadata

    assert isinstance(
        metadata, dict
    ), f"Expected metadata to be dict, got {type(metadata)}"

    if "loss" in kwargs:
        loss = kwargs["loss"]
        kwargs.pop("loss")
    else:
        loss = SMAPE()

    net = estimator_cls(
        metadata=metadata,
        loss=loss,
        **kwargs,
    )

    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    test_outputs = trainer.test(net, dataloaders=test_dataloader)
    assert len(test_outputs) > 0

    # todo: add the predict pipeline and make this test cleaner
    x, y = next(iter(test_dataloader))
    net.eval()
    with torch.no_grad():
        output = net(x)
    net.train()
    prediction = output["prediction"]
    n_dims = prediction.ndim
    assert n_dims == 3, (
        f"Prediction output must be 3D, but got {n_dims}D tensor "
        f"with shape {output.shape}"
    )

    shutil.rmtree(tmp_path, ignore_errors=True)


class TestAllPtForecastersV2(EstimatorPackageConfig, EstimatorFixtureGenerator):
    """Generic tests for all objects in the mini package."""

    object_type_filter = "forecaster_pytorch_v2"

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
        object_class = object_pkg.get_cls()
        dataloaders = object_pkg._get_test_datamodule_from(trainer_kwargs)

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
        assert object_pkg.__name__ == object_class.__name__ + "_pkg_v2", msg

    def test_d2_metadata(self, object_pkg, trainer_kwargs):
        object_class = object_pkg.get_cls()
        dataloaders = object_pkg._get_test_datamodule_from(trainer_kwargs)
        data_module = dataloaders.get("data_module")
        metadata = data_module.metadata

        model_kwargs = dict(trainer_kwargs)
        model_kwargs.pop("data_loader_kwargs", None)

        model_name = object_class.__name__

        check_method_name = f"_check_metadata_{model_name.lower()}"
        if hasattr(object_pkg, check_method_name):
            getattr(object_pkg, check_method_name)(metadata)
