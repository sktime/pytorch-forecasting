"""
Packages container for TSMixer model.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class TSMixer_pkg_v2(Base_pkg):
    """TSMixer package container."""

    _tags = {
        "info:name": "TSMixer",
        "info:compute": 2,
        "authors": ["seaic-mac-murchadha"],
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.tsmixer._tsmixer_v2 import TSMixer

        return TSMixer

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def get_test_train_params(cls):
        """
        Return testing parameter settings for the trainer.

        Returns
        -------
        list[dict]
            Parameter configurations used to create testing instances of the TSMixer class.
        """

        import torch.nn as nn

        from pytorch_forecasting.metrics import SMAPE

        params = [
            {},
            dict(
                d_model=64,
                e_layers=2,
                dropout=0.1,
                logging_metrics=[SMAPE()],
            ),
            dict(
                d_model=32,
                e_layers=1,
                dropout=0.0,
                loss=nn.MSELoss(),
            ),
            dict(
                optimizer="adamw",
                optimizer_params={"lr": 1e-3},
            ),
        ]

        default_dm_cfg = {
            "context_length": 8,
            "prediction_length": 2,
        }

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            param["datamodule_cfg"] = {
                **default_dm_cfg,
                **current_dm_cfg,
            }

        return params