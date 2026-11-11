"""NBeats v2 package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class NBeats_pkg_v2(Base_pkg):
    """NBeats v2 package container."""

    _tags = {
        "info:name": "NBeats",
        "info:compute": 1,
        "info:y_type": ["numeric"],
        "authors": [
            "dmitri-carpov"  # paper author
            "jdb78",  # for v1
            "Faakhir30",
        ],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.nbeats._nbeats_v2 import NBeats

        return NBeats

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    @classmethod
    def get_test_train_params(cls):
        """Return testing parameter settings for the trainer."""
        from pytorch_forecasting.metrics import MAE, MAPE, SMAPE

        params = [
            {
                "widths": [16, 32],
                "num_blocks": [1, 1],
                "num_block_layers": [2, 2],
            },
            {
                "backcast_loss_ratio": 1.0,
                "widths": [16, 32],
                "num_blocks": [1, 1],
                "num_block_layers": [2, 2],
            },
            {
                "stack_types": ["generic"],
                "num_blocks": [1],
                "num_block_layers": [2],
                "widths": [16],
                "expansion_coefficient_lengths": [8],
                "sharing": [False],
            },
            {
                "loss": MAE(),
                "widths": [16, 32],
                "num_blocks": [1, 1],
                "num_block_layers": [2, 2],
            },
            {
                "loss": MAPE(),
                "logging_metrics": [SMAPE()],
                "widths": [16, 32],
                "num_blocks": [1, 1],
                "num_block_layers": [2, 2],
            },
            {
                "optimizer": "adamw",
                "lr_scheduler": "cosine_annealing",
                "lr_scheduler_params": {"T_max": 5},
                "widths": [16, 32],
                "num_blocks": [1, 1],
                "num_block_layers": [2, 2],
            },
        ]

        default_dm_cfg = {
            "max_encoder_length": 8,
            "max_prediction_length": 3,
            "min_encoder_length": 8,
            "min_prediction_length": 3,
            "add_relative_time_idx": False,
        }

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = default_dm_cfg.copy()

        return params
