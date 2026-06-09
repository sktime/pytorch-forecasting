"""
Packages container for Autoformer model.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class Autoformer_pkg_v2(Base_pkg):
    """Autoformer package container."""

    _tags = {
        "info:name": "Autoformer",
        "info:compute": 2,
        "authors": ["harshsomankar123-tech"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.autoformer._autoformer_v2 import Autoformer

        return Autoformer

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def get_test_train_params(cls):
        """
        Return testing parameter settings for the trainer.
        """
        from pytorch_forecasting.metrics import SMAPE

        params = [
            # First set: smaller network params for fast testing
            dict(
                hidden_size=16,
                n_heads=2,
                e_layers=1,
                d_layers=1,
                d_ff=32,
            ),
            # Second set: custom moving_avg and logging metrics
            dict(
                hidden_size=8,
                n_heads=2,
                e_layers=1,
                d_layers=1,
                d_ff=16,
                moving_avg=5,
                logging_metrics=[SMAPE()],
            ),
            # Third set: custom scheduler
            dict(
                hidden_size=8,
                n_heads=2,
                e_layers=1,
                d_layers=1,
                d_ff=16,
                optimizer="adamw",
                lr_scheduler="cosine_annealing",
                lr_scheduler_params={"T_max": 5},
            ),
        ]

        default_dm_cfg = {"context_length": 8, "prediction_length": 2}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)

            param["datamodule_cfg"] = default_dm_cfg

        return params
