"""
DecoderMLP v2 package container.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class DecoderMLP_pkg_v2(Base_pkg):
    """DecoderMLP v2 package container."""

    _tags = {
        "info:name": "DecoderMLP",
        "info:compute": 1,
        "authors": ["jdb78", "amaydixit11"],
        "capability:exogenous": True,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:flexible_history_length": True,
        "capability:cold_start": True,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.mlp._decodermlp_v2 import DecoderMLP

        return DecoderMLP

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    @classmethod
    def get_test_train_params(cls):
        """Return testing parameter settings for the trainer.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        params = [
            {},
            dict(
                activation_class="ReLU",
                hidden_size=32,
                n_hidden_layers=2,
                dropout=0.1,
                norm=True,
            ),
            dict(
                activation_class="GELU",
                hidden_size=16,
                n_hidden_layers=1,
                dropout=0.0,
                norm=False,
            ),
        ]

        default_dm_cfg = {
            "max_encoder_length": 6,
            "max_prediction_length": 3,
        }

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            dm_cfg = default_dm_cfg.copy()
            dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = dm_cfg

        return params
