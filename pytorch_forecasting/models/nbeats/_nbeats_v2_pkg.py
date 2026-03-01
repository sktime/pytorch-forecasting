from pytorch_forecasting.base._base_pkg import Base_pkg


class NBEATS_pkg_v2(Base_pkg):
    """NBEATS v2 package container."""

    _tags = {
        "info:name": "NBEATS",
        "authors": ["PalakB09"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.nbeats._nbeats_v2 import NBEATS

        return NBEATS

    @classmethod
    def get_datamodule_cls(cls):
        """Get datamodule class used for training."""
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
            {
                "stack_types": ["trend", "seasonality"],
                "num_blocks": [2, 2],
                "num_block_layers": [2, 2],
                "widths": [16, 32],
                "sharing": [True, True],
                "expansion_coefficient_lengths": [3, 7],
            },
            {
                "stack_types": ["generic"],
                "num_blocks": [2],
                "num_block_layers": [2],
                "widths": [32],
                "sharing": [False],
                "expansion_coefficient_lengths": [16],
            },
        ]

        default_dm_cfg = {
            "max_encoder_length": 6,
            "max_prediction_length": 4,
        }

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            dm_cfg = default_dm_cfg.copy()
            dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = dm_cfg

        return params
