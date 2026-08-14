from pathlib import Path
import shutil
from typing import Any

import torch
import yaml

from pytorch_forecasting.base._base_pkg import Base_pkg
from pytorch_forecasting.data import TimeSeries


def _integration(
    pkg: Base_pkg,
    test_data: dict[str, TimeSeries],
    datamodule_cfg: dict[str, Any],
    **kwargs,
):
    """Test integration of models with the `TimeSeries` and datamodules"""
    pkg.fit(test_data["train"])

    predictions = pkg.predict(
        test_data["predict"],
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


def _check_save(ckpt_dir, best_model_path):
    """Test saving artifacts.

    Checks:

    - if the artifacts are stored correctly
    - artifacts.yaml exists and has correct format
    """
    artifacts_yaml_path = ckpt_dir / "artifacts.yaml"
    assert artifacts_yaml_path.is_file(), "artifacts.yaml not created"

    with open(artifacts_yaml_path) as f:
        artifacts_data = yaml.safe_load(f)
    artifacts = artifacts_data["artifacts"]

    # All mandatory keys must be present
    for cfg_key in (
        "model_cfg",
        "datamodule_cfg",
        "trainer_cfg",
        "datamodule_metadata",
    ):
        assert cfg_key in artifacts, f"'{cfg_key}' not recorded in artifacts.yaml"

    # Are all the mandatory keys saved as file
    model_cfg = Path(artifacts["model_cfg"])
    datamodule_cfg = Path(artifacts["datamodule_cfg"])
    trainer_cfg = Path(artifacts["trainer_cfg"])
    metadata = Path(artifacts["datamodule_metadata"])
    assert model_cfg.is_file(), "model_cfg.pkl not saved"
    assert datamodule_cfg.is_file(), "datamodule_cfg.pkl not saved"
    assert trainer_cfg.is_file(), "trainer_cfg.pkl not saved"
    assert metadata.is_file(), "datamodule_metadata.pkl not saved"

    # Model checkpoint must be present
    assert (
        "best_model_checkpoint" in artifacts
    ), "'best_model_checkpoint' not recorded in artifacts.yaml"

    recorded_ckpt_path = Path(artifacts["best_model_checkpoint"])
    assert (
        recorded_ckpt_path.is_file()
    ), f"Recorded checkpoint does not exist at {recorded_ckpt_path}"
    assert (
        recorded_ckpt_path == best_model_path
    ), "Recorded checkpoint path in artifacts.yaml doesn't match path returned by fit()"


def _check_load(test_data, pkg_loaded, tmp_path):
    """Test loading artifacts.

    Checks:
    - if the artifacts are loaded correctly
    - if the model is instantiated from checkpoint and loaded correctly
    """
    assert pkg_loaded.model is not None, "Model was not loaded from checkpoint"

    # Configs loaded from pkl files
    assert pkg_loaded.model_cfg != {}, "model_cfg was not loaded"
    assert pkg_loaded.datamodule_cfg != {}, "datamodule_cfg was not loaded"
    assert pkg_loaded.trainer_cfg != {}, "trainer_cfg was not loaded"

    # metadata, trainer, model is loaded correctly
    assert pkg_loaded.metadata != {}, "metadata was not loaded"
    assert pkg_loaded.trainer is not None, "Trainer was not instantiated from cfg"
    assert pkg_loaded.model is not None, "Model was not instantiated from checkpoint."

    shutil.rmtree(tmp_path, ignore_errors=True)
