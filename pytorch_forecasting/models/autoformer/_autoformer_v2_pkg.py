"""
Autoformer package container for v2 interface.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class Autoformer_pkg_v2(Base_pkg):
    """Autoformer package container."""

    _tags = {
        "info:name": "Autoformer",
        "authors": ["Satarupa22-SD"],
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.autoformer._autoformer_v2 import Autoformer

        return Autoformer

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    @classmethod
    def get_test_train_params(cls):
        """Return testing parameters settings for the trainer.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
            Each dict contains parameters used to construct an
            "interesting" test instance, i.e., `MyClass(**params)`
            or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only)
            dictionary in `params`.
        """
        from pytorch_forecasting.metrics import QuantileLoss

        params = [
            {
                "d_model": 32,
                "enc_layers": 1,
                "dec_layers": 1,
                "moving_avg": 25,
                "use_revin": False,
            },
            {
                "d_model": 16,
                "enc_layers": 1,
                "dec_layers": 1,
                "moving_avg": 25,
                "use_revin": True,
                "out_channels": 1,
            },
            {
                "loss": QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
                "d_model": 32,
                "enc_layers": 2,
                "dec_layers": 1,
                "moving_avg": 25,
                "use_revin": False,
            },
        ]

        default_dm_cfg = {"max_encoder_length": 4, "max_prediction_length": 3}

        for p in params:
            p["datamodule_cfg"] = default_dm_cfg

        return params
