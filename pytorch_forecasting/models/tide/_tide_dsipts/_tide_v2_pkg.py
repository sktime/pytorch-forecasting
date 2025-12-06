"""TIDE package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class TIDE_pkg_v2(Base_pkg):
    """TIDE package container."""

    _tags = {
        "info:name": "TIDE",
        "authors": ["fbk_dsipts"],
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.tide._tide_dsipts._tide_v2 import TIDE

        return TIDE

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from pytorch_forecasting.metrics import MAE, MAPE

        params = [
            dict(
                hidden_size=16,
                d_model=8,
                n_add_enc=1,
                n_add_dec=1,
                dropout_rate=0.1,
            ),
            dict(
                hidden_size=32,
                d_model=16,
                n_add_enc=2,
                n_add_dec=2,
                dropout_rate=0.2,
                datamodule_cfg=dict(max_encoder_length=5, max_prediction_length=3),
                loss=MAE(),
            ),
            dict(
                hidden_size=64,
                d_model=32,
                n_add_enc=3,
                n_add_dec=2,
                dropout_rate=0.1,
                datamodule_cfg=dict(max_encoder_length=4, max_prediction_length=2),
                loss=MAPE(),
            ),
        ]
        default_dm_cfg = {"max_encoder_length": 4, "max_prediction_length": 3}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)

            param["datamodule_cfg"] = default_dm_cfg

        return params
