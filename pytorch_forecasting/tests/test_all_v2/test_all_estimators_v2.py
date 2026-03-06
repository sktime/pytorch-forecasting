"""Automated tests based on the skbase test suite template."""

import os
from pathlib import Path
import shutil

import torch

from pytorch_forecasting.tests.test_all_estimators import (
    EstimatorFixtureGenerator,
    EstimatorPackageConfig,
)
from pytorch_forecasting.tests.test_all_v2._test_integration import _integration
from pytorch_forecasting.tests.test_all_v2.utils import _setup_pkg_and_data

# whether to test only estimators from modules that are changed w.r.t. main
# default is False, can be set to True by pytest --only_changed_modules True flag
ONLY_CHANGED_MODULES = False


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
        pkg, test_data, dm_cfg = _setup_pkg_and_data(
            object_pkg, trainer_kwargs, tmp_path
        )

        _integration(pkg, test_data, dm_cfg)

        shutil.rmtree(tmp_path, ignore_errors=True)

    def test_checkpointing(self, object_pkg, trainer_kwargs, tmp_path):
        """Test that the package can save a checkpoint and reload from it."""
        pkg, test_data, _ = _setup_pkg_and_data(object_pkg, trainer_kwargs, tmp_path)

        ckpt_dir = Path(tmp_path) / "checkpoints"
        best_model_path = pkg.fit(
            test_data["train"],
            save_ckpt=True,
            ckpt_dir=ckpt_dir,
            ckpt_kwargs={"monitor": "train_loss_epoch"},
        )

        assert best_model_path is not None
        assert os.path.exists(best_model_path)

        dm_cfg_path = Path(best_model_path).parent / "model_cfg.pkl"
        assert (
            dm_cfg_path.exists()
        ), "datamodule_cfg.pkl was not saved alongside checkpoint"

        pkg_loaded = object_pkg(ckpt_path=best_model_path)

        predictions = pkg_loaded.predict(test_data["predict"], mode="prediction")

        assert predictions is not None
        assert "prediction" in predictions
        shutil.rmtree(tmp_path, ignore_errors=True)

    def test_predict_modes(self, object_pkg, trainer_kwargs, tmp_path):
        """Test different prediction modes and return_info."""
        pkg, test_data, _ = _setup_pkg_and_data(object_pkg, trainer_kwargs, tmp_path)

        pkg.fit(test_data["train"], save_ckpt=False)
        predict_data = test_data["predict"]

        # mode="raw"
        raw_out = pkg.predict(predict_data, mode="raw")
        raw_pred_tensor = raw_out["prediction"]
        assert any(isinstance(v, torch.Tensor) for v in raw_out.values())
        assert (
            raw_pred_tensor.ndim == 3
        ), f"Prediction must be 3D, got {raw_pred_tensor.ndim}D"

        # mode="quantiles"
        quantile_out = pkg.predict(predict_data, mode="quantiles")
        quanitle_pred_tensor = quantile_out["prediction"]
        assert isinstance(quanitle_pred_tensor, torch.Tensor)
        assert (
            quanitle_pred_tensor.ndim == 3
        ), f"Prediction must be 3D, got {quanitle_pred_tensor.ndim}D"

        # mode="prediction"
        pred_out = pkg.predict(predict_data, mode="prediction")
        pred_tensor = pred_out["prediction"]
        assert isinstance(pred_tensor, torch.Tensor)
        assert pred_tensor.ndim == 2, f"Prediction must be 3D, got {pred_tensor.ndim}D"

        return_info_keys = ["index", "x"]
        info_out = pkg.predict(
            predict_data, mode="prediction", return_info=return_info_keys
        )

        for key in return_info_keys:
            assert key in info_out, f"Requested key '{key}' missing from output"

        assert info_out["index"] is not None
        assert isinstance(info_out["x"], dict)

        shutil.rmtree(tmp_path, ignore_errors=True)

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
        class_name = object_class.__name__

        expected_names = {class_name + "_pkg_v2"}

        if class_name.endswith("_v2"):
            expected_names.add(class_name[:-3] + "_pkg_v2")

        msg = (
            f"Package class '{object_pkg.__name__}' does not follow the expected "
            f"naming convention for estimator '{class_name}'. "
            f"Expected one of: {sorted(expected_names)}."
        )

        assert object_pkg.__name__ in expected_names, msg
