"""xLSTMTime v2 package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class xLSTMTime_pkg_v2(Base_pkg):
    """xLSTMTime v2 package container."""

    _tags = {
        "info:name": "xLSTMTime",
        "info:compute": 3,
        "info:y_type": ["numeric"],
        "authors": ["muslehal", "phoeenniixx", "Faakhir30"],
        "capability:exogenous": True,
        "capability:multivariate": False,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.xlstm._xlstm_v2 import xLSTMTime_v2

        return xLSTMTime_v2

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
        from pytorch_forecasting.metrics import MAE, MAPE, QuantileLoss

        params = [
            {},
            {"xlstm_type": "mlstm"},
            {"num_layers": 2},
            {"xlstm_type": "slstm", "input_projection_size": 32},
            {
                "xlstm_type": "mlstm",
                "decomposition_kernel": 3,
                "dropout": 0.2,
                "loss": MAE(),
            },
            {
                "loss": QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
                "hidden_size": 16,
            },
            {
                "optimizer": "adamw",
                "lr_scheduler": "cosine_annealing",
                "lr_scheduler_params": {"T_max": 5},
            },
        ]

        default_dm_cfg = {
            "max_encoder_length": 8,
            "max_prediction_length": 3,
        }

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = default_dm_cfg.copy()

        return params
