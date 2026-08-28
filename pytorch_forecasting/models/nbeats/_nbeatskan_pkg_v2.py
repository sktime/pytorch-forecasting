"""NBeatsKAN v2 package container."""

from typing import Any

from pytorch_forecasting.base._base_pkg import Base_pkg


class NBeatsKAN_pkg_v2(Base_pkg):
    """NBeatsKAN v2 package container."""

    _tags: dict[str, Any] = {
        "info:name": "NBeatsKAN",
        "info:compute": 2,
        "info:y_type": ["numeric"],
        "authors": ["jdb78", "Dev10-sys"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get the model class."""
        from pytorch_forecasting.models.nbeats._nbeatskan_v2 import NBeatsKAN_v2

        return NBeatsKAN_v2

    @classmethod
    def get_datamodule_cls(cls):
        """Get the compatible DataModule class."""
        from pytorch_forecasting.data.data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def get_test_train_params(cls) -> list[dict[str, Any]]:
        """Return testing parameter settings for trainer fixtures."""
        from pytorch_forecasting.metrics import MAE

        params: list[dict[str, Any]] = [
            {
                "widths": [16, 32],
                "num_blocks": [1, 1],
                "num_block_layers": [2, 2],
            },
            {
                "backcast_loss_ratio": 1.0,
                "widths": [16, 32],
                "num_blocks": [1, 1],
                "num_block_layers": [2, 2],
            },
            {
                "stack_types": ["generic"],
                "num_blocks": [1],
                "num_block_layers": [2],
                "widths": [16],
                "expansion_coefficient_lengths": [8],
                "sharing": [False],
            },
            {
                "loss": MAE(),
                "widths": [16, 32],
                "num_blocks": [1, 1],
                "num_block_layers": [2, 2],
            },
        ]

        default_dm_cfg: dict[str, Any] = {
            "context_length": 8,
            "prediction_length": 3,
            "add_relative_time_idx": False,
        }

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = default_dm_cfg.copy()

        return params
