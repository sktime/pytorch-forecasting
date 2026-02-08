from pytorch_forecasting.base._base_pkg import Base_pkg


class NBEATS_pkg_v2(Base_pkg):
    _tags = {
        "info:name": "NBEATS",
        "authors": ["PalakB09"],
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.models.nbeats._nbeats_v2 import NBEATS

        return NBEATS

    @classmethod
    def get_datamodule_cls(cls):
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    @classmethod
    def get_test_train_params(cls):
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
