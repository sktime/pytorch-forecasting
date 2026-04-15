"""
Metadata container for Moirai-MoE v2.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class MoiraiMoE_pkg_v2(Base_pkg):
    """Moirai-MoE metadata container."""

    _tags = {
        "info:name": "MoiraiMoE",
        "authors": ["priyanshuharshbodhi1"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "python_dependencies": ["uni2ts", "torch", "einops", "huggingface_hub"],
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.moirai_moe._moirai_moe_v2 import MoiraiMoE

        return MoiraiMoE

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def get_test_train_params(cls):
        """Return testing parameter settings for the trainer.

        The default checkpoint ``Salesforce/moirai-moe-1.0-R-small`` is kept
        across configurations to avoid re-downloading larger backbones during
        CI runs. ``training=False`` freezes the backbone so gradients are not
        tracked through the pretrained weights in the smoke tests.
        """
        params = [
            dict(training=False, num_samples=20),
            dict(
                training=False,
                patch_size=16,
                num_samples=10,
                datamodule_cfg=dict(context_length=32, prediction_length=8),
            ),
        ]

        default_dm_cfg = {"context_length": 32, "prediction_length": 8}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            merged = {**default_dm_cfg, **current_dm_cfg}
            param["datamodule_cfg"] = merged

        return params
