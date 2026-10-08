"""Baseline v2 package container."""

from typing import Any

from pytorch_forecasting.base._base_pkg import Base_pkg


class Baseline_pkg_v2(Base_pkg):
    """Baseline v2 package container."""

    _tags: dict[str, Any] = {
        "info:name": "Baseline",
        "info:compute": 1,
        "info:y_type": ["numeric"],
        "authors": ["jdb78", "Dev10-sys"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get the model class."""
        from pytorch_forecasting.models.baseline._baseline_v2 import Baseline_v2

        return Baseline_v2

    @classmethod
    def get_datamodule_cls(cls):
        """Get the compatible DataModule class."""
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    @classmethod
    def get_test_train_params(cls) -> list[dict[str, Any]]:
        """Return testing parameter settings for trainer fixtures."""
        from pytorch_forecasting.metrics import MAE, RMSE, SMAPE

        params: list[dict[str, Any]] = [
            {},
            dict(loss=RMSE()),
            dict(loss=SMAPE(), logging_metrics=[MAE()]),
        ]

        default_dm_cfg: dict[str, Any] = {
            "max_encoder_length": 8,
            "max_prediction_length": 3,
        }

        for param in params:
            dm_cfg = default_dm_cfg.copy()
            dm_cfg.update(param.get("datamodule_cfg", {}))
            param["datamodule_cfg"] = dm_cfg

        return params
