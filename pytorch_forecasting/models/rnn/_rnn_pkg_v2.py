"""RecurrentNetwork v2 package container."""

from typing import Any

from pytorch_forecasting.base._base_pkg import Base_pkg


class RecurrentNetwork_pkg_v2(Base_pkg):
    """RecurrentNetwork v2 package container."""

    _tags: dict[str, Any] = {
        "info:name": "RecurrentNetwork",
        "info:compute": 1,
        "info:y_type": ["numeric"],
        "authors": ["jdb78", "Dev10-sys"],
        "capability:exogenous": True,
        "capability:multivariate": False,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get the model class."""
        from pytorch_forecasting.models.rnn._rnn_v2 import RecurrentNetwork_v2

        return RecurrentNetwork_v2

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
        from pytorch_forecasting.metrics import MAE, QuantileLoss

        params: list[dict[str, Any]] = [
            {},
            {"cell_type": "GRU"},
            {"rnn_layers": 2, "dropout": 0.2},
            {"hidden_size": 8, "cell_type": "LSTM"},
            {"loss": MAE()},
            {
                "loss": QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
                "hidden_size": 16,
            },
        ]

        default_dm_cfg: dict[str, Any] = {
            "max_encoder_length": 8,
            "max_prediction_length": 3,
        }

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            if isinstance(current_dm_cfg, dict):
                default_dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = default_dm_cfg.copy()

        return params
