"""NHiTS v2 package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class NHiTS_v2_pkg_v2(Base_pkg):
    """NHiTS v2 package container."""

    _tags = {
        "info:name": "NHiTS_v2",
        "info:compute": 2,
        "authors": ["echo-xiao"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class.

        Returns
        -------
        NHiTS_v2 : type
            The model class.
        """
        from pytorch_forecasting.models.nhits._nhits_v2 import NHiTS_v2

        return NHiTS_v2

    @classmethod
    def get_datamodule_cls(cls):
        """Get datamodule class used for training.

        Returns
        -------
        EncoderDecoderTimeSeriesDataModule : type
            The datamodule class.
        """
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
            Each dict is passed as ``model_cfg`` to the package constructor.
            The key ``"datamodule_cfg"`` inside each dict is forwarded to
            the datamodule constructor.
        """
        from pytorch_forecasting.metrics import MAE, SMAPE

        params = [
            {},
            dict(
                n_blocks=[1, 1, 1],
                n_layers=2,
                hidden_size=256,
                pooling_mode="max",
                interpolation_mode="linear",
            ),
            dict(
                n_blocks=[1, 1],
                hidden_size=128,
                backcast_loss_ratio=0.1,
                logging_metrics=[SMAPE()],
            ),
            dict(
                n_blocks=[1],
                hidden_size=64,
                loss=MAE(),
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
