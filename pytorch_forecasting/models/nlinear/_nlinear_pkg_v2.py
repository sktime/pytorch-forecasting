"""
Packages container for NLinear model.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class NLinear_pkg_v2(Base_pkg):
    """NLinear package container."""

    _tags = {
        "info:name": "NLinear",
        "info:compute": 2,
        "authors": ["mixiancmx", "Sylver.Icy"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.nlinear._nlinear_v2 import NLinear

        return NLinear

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def get_test_train_params(cls):
        """Return testing parameter settings for the trainer."""
        params = [
            {},
            dict(individual=False),
            dict(individual=True),
            dict(
                individual=False,
                datamodule_cfg=dict(context_length=12, prediction_length=3),
            ),
            dict(
                individual=True,
                datamodule_cfg=dict(context_length=16, prediction_length=4),
            ),
        ]

        default_dm_cfg = {"context_length": 8, "prediction_length": 2}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = default_dm_cfg.copy()

        return params
