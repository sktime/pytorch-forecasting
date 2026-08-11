"""ModernTCN package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class ModernTCN_pkg_v2(Base_pkg):
    """
    ModernTCN package container

    GitHub Repository:https://github.com/luodhhh/ModernTCN

    Research Paper: https://openreview.net/forum?id=vpJMJerXHU

    """

    _tags = {
        "info:name": "ModernTCN",
        "authors": ["Muhammad-Rebaal", "luodhhh"],
        "info:compute": 2,
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.modern_tcn._modern_tcn_v2 import ModernTCN

        return ModernTCN

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    @classmethod
    def get_test_train_params(cls):
        """Return testing parameter settings."""
        from pytorch_forecasting.metrics import QuantileLoss

        params = [
            {},
            {
                "d_model": 16,
                "kernel_size": 3,
                "n_blocks": 1,
                "d_ff": 32,
                "patch_size": 4,
                "use_revin": False,
            },
            {
                "d_model": 8,
                "kernel_size": 3,
                "n_blocks": 1,
                "d_ff": 16,
                "patch_size": 2,
                "use_revin": True,
            },
            {
                "loss": QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
                "d_model": 16,
                "kernel_size": 3,
                "n_blocks": 1,
                "d_ff": 32,
                "patch_size": 4,
                "use_revin": False,
            },
            {
                "d_model": 16,
                "kernel_size": 7,
                "small_kernel_size": 3,
                "n_blocks": 1,
                "d_ff": 32,
                "patch_size": 4,
                "individual": True,
                "use_revin": False,
            },
        ]

        for param in params:
            dm_cfg = {"max_encoder_length": 8, "max_prediction_length": 2}
            dm_cfg.update(param.get("datamodule_cfg", {}))
            param["datamodule_cfg"] = dm_cfg

        return params
