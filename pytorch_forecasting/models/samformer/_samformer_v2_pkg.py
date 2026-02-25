"""
Samformer package container.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class Samformer_pkg_v2(Base_pkg):
    """Samformer package container."""

    _tags = {
        "info:name": "Samformer",
        "authors": ["fbk_dsipts"],
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.samformer._samformer_v2 import Samformer

        return Samformer

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
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        import torch.nn as nn

        from pytorch_forecasting.metrics import QuantileLoss

        params = [
            {
                # "loss": nn.MSELoss(),
                "hidden_size": 32,
                "use_revin": False,
            },
            {
                # "loss": nn.MSELoss(),
                "hidden_size": 16,
                "use_revin": True,
                "out_channels": 1,
                "persistence_weight": 0.0,
            },
            {
                "loss": QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
                "hidden_size": 32,
                "use_revin": False,
            },
        ]

        default_dm_cfg = {"max_encoder_length": 4, "max_prediction_length": 3}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)

            param["datamodule_cfg"] = default_dm_cfg

        return params
