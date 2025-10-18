"""Automated tests based on the skbase test suite template."""

import shutil
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.nn as nn

from pytorch_forecasting.base._base_pkg import Base_pkg
from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.tests.test_all_estimators import (
    EstimatorFixtureGenerator,
    EstimatorPackageConfig,
)

# whether to test only estimators from modules that are changed w.r.t. main
# default is False, can be set to True by pytest --only_changed_modules True flag
ONLY_CHANGED_MODULES = False


def _integration(
    estimator_cls: type[Base_pkg],
    test_data: dict[str, TimeSeries],
    model_cfg: dict[str, Any],
    datamodule_cfg: dict[str, Any],
    tmp_path: str,
    trainer_cfg: Optional[dict[str, Any]] = None,
    **kwargs,
):
    train_data = test_data["train"]
    predict_data = test_data["predict"]

    default_model_cfg = {"loss": SMAPE()}

    default_datamodule_cfg = {
        "train_val_test_split": (0.8, 0.2),
        "add_relative_time_idx": True,
        "batch_size": 2,
    }

    logger = TensorBoardLogger(tmp_path)
    default_trainer_cfg = {
        "max_epochs": 3,
        "gradient_clip_val": 0.1,
        "enable_checkpointing": True,
        "default_root_dir": tmp_path,
        "limit_train_batches": 2,
        "limit_val_batches": 2,
        "logger": logger,
    }
    default_model_cfg.update(model_cfg)
    default_datamodule_cfg.update(datamodule_cfg)
    if trainer_cfg is not None:
        default_trainer_cfg.update(trainer_cfg)

    pkg = estimator_cls(
        model_cfg=default_model_cfg,
        trainer_cfg=default_trainer_cfg,
        datamodule_cfg=default_datamodule_cfg,
    )

    pkg.fit(train_data)

    predictions = pkg.predict(
        predict_data,
        mode="raw",
    )

    assert predictions is not None
    assert isinstance(predictions, dict)
    assert "prediction" in predictions

    pred_tensor = predictions["prediction"]
    assert isinstance(pred_tensor, torch.Tensor)
    assert pred_tensor.ndim == 3, f"Prediction must be 3D, got {pred_tensor.ndim}D"

    expected_pred_len = datamodule_cfg.get("prediction_length")
    if expected_pred_len:
        assert pred_tensor.shape[1] == expected_pred_len, (
            f"Pred length mismatch: expected {expected_pred_len}, "
            f"got {pred_tensor.shape[1]}"
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
        params_copy = trainer_kwargs.copy()
        datamodule_cfg = params_copy.pop("datamodule_cfg", {})
        model_cfg = params_copy

        test_data = object_pkg.get_test_data(**datamodule_cfg)

        _integration(
            estimator_cls=object_pkg,
            test_data=test_data,
            model_cfg=model_cfg,
            datamodule_cfg=datamodule_cfg,
            tmp_path=tmp_path,
        )

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
