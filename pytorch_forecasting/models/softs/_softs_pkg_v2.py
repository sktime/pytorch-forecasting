"""
Packages container for SOFTS model.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class SOFTS_pkg_v2(Base_pkg):
    """
    SOFTS package container.
    Reference : https://arxiv.org/abs/2404.14197
    """

    _tags = {
        "info:name": "SOFTS",
        "info:y_type": ["numeric"],
        "info:compute": 2,
        "authors": ["Secilia-Cxy", "Muhammad-Rebaal"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.softs._softs_v2 import SOFTS

        return SOFTS

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
        list of dict
            Each dict is a valid set of constructor arguments for ``SOFTS``.
            The key ``datamodule_cfg`` is passed to the DataModule, not the model.
        """
        from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE

        params = [
            {},
            dict(hidden_size=64, d_core=64, d_ff=256, n_layers=1),
            dict(hidden_size=128, n_layers=1, use_revin=False),
            dict(
                hidden_size=64,
                n_layers=1,
                loss=MAE(),
            ),
            dict(
                hidden_size=64,
                n_layers=1,
                loss=MAPE(),
            ),
            dict(
                hidden_size=64,
                n_layers=1,
                loss=RMSE(),
            ),
            dict(
                hidden_size=64,
                n_layers=1,
                use_revin=False,
                loss=MAE(),
            ),
            dict(hidden_size=64, dropout=0.0, n_layers=1),
            dict(datamodule_cfg=dict(max_encoder_length=16, max_prediction_length=4)),
            dict(
                optimizer="adamw",
                lr_scheduler="cosine_annealing",
                lr_scheduler_params={"T_max": 5},
            ),
            dict(
                optimizer="adagrad",
                optimizer_params={"lr": 1e-3},
            ),
            dict(hidden_size=64, n_layers=1, logging_metrics=[SMAPE()]),
        ]

        default_dm_cfg = {"max_encoder_length": 8, "max_prediction_length": 2}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            param["datamodule_cfg"] = {**default_dm_cfg, **current_dm_cfg}

        return params
