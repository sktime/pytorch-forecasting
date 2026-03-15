"""
Package container for the PatchTST model.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class PatchTST_pkg_v2(Base_pkg):
    """PatchTST package container."""

    _tags = {
        "info:name": "PatchTST",
        "info:compute": 3,
        "authors": ["amruth6002"],
        "capability:exogenous": False,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.patchtst._patchtst_v2 import PatchTST

        return PatchTST

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def get_test_train_params(cls):
        """
        Return testing parameter settings for the trainer.

        Parameters
        ----------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss

        params = [
            {},  # defaults (d_model=128, n_heads=16, e_layers=3)
            dict(d_model=16, n_heads=4, e_layers=1, d_ff=32, patch_len=4, stride=2),
            dict(
                d_model=32,
                n_heads=4,
                e_layers=2,
                d_ff=64,
                patch_len=8,
                stride=4,
                logging_metrics=[SMAPE()],
            ),
            dict(
                d_model=16,
                n_heads=4,
                e_layers=1,
                d_ff=32,
                patch_len=4,
                stride=2,
                loss=QuantileLoss(),
            ),
        ]

        default_dm_cfg = {"context_length": 16, "prediction_length": 2}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = default_dm_cfg

        return params
