"""
Packages container for Baseline model.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class Baseline_pkg_v2(Base_pkg):
    """Baseline package container."""

    _tags = {
        "info:name": "Baseline_v2",
        "info:compute": 1,
        "authors": ["Utkarshhs"],
        "info:y_type": ["numeric"],
        "capability:exogenous": False,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.baseline._baseline_v2 import Baseline_v2

        return Baseline_v2

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def get_test_train_params(cls):
        import torch.nn as nn

        from pytorch_forecasting.metrics import SMAPE

        params = [
            {},
            dict(logging_metrics=[SMAPE()]),
            dict(
                loss=nn.MSELoss(),
                logging_metrics=[SMAPE()],
            ),
            dict(
                loss=nn.HuberLoss(),
            ),
        ]

        default_dm_cfg = {"context_length": 8, "prediction_length": 2}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            dm_cfg = default_dm_cfg.copy()
            default_dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = dm_cfg

        return params
