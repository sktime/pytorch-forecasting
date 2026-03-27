"""FreTS v2 package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class FreTS_v2_pkg_v2(Base_pkg):
    """FreTS v2 package container."""

    _tags = {
        "info:name": "FreTS_v2",
        "info:compute": 2,
        "authors": ["echo-xiao"],
        "capability:exogenous": False,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class.

        Returns
        -------
        FreTS_v2 : type
            The model class.
        """
        from pytorch_forecasting.models.frets._frets_v2 import FreTS_v2

        return FreTS_v2

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
                embed_size=32,
                hidden_size=64,
                channel_independence=True,
            ),
            dict(
                embed_size=64,
                hidden_size=128,
                channel_independence=False,
                logging_metrics=[SMAPE()],
            ),
            dict(
                embed_size=16,
                hidden_size=32,
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
