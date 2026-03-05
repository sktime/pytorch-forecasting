"""
Package container for RecurrentNetwork v2 model.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class RecurrentNetwork_pkg_v2(Base_pkg):
    """RecurrentNetwork v2 package container."""

    _tags = {
        "info:name": "RecurrentNetwork_v2",
        "info:compute": 2,
        "authors": ["Meet-Ramjiyani-10"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.rnn._rnn_v2 import RecurrentNetwork_v2

        return RecurrentNetwork_v2

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    @classmethod
    def get_test_train_params(cls):
        """
        Return testing parameter settings for the trainer.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss

        params = [
            dict(loss=MAE()),
            dict(
                loss=SMAPE(),
                cell_type="LSTM",
                hidden_size=32,
                rnn_layers=1,
            ),
            dict(
                loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
                cell_type="GRU",
                hidden_size=32,
                rnn_layers=1,
            ),
            dict(
                loss=MAE(),
                cell_type="LSTM",
                hidden_size=16,
                rnn_layers=2,
                dropout=0.1,
            ),
        ]

        default_dm_cfg = {"context_length": 8, "prediction_length": 2}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = default_dm_cfg

        return params
