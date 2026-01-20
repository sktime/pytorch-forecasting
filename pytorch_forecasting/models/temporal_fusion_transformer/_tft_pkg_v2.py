"""TFT package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class TFT_pkg_v2(Base_pkg):
    """TFT package container."""

    _tags = {
        "info:name": "TFT",
        "authors": ["phoeenniixx"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import TFT

        return TFT

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
        params = [
            {},
            dict(
                hidden_size=25,
                attention_head_size=5,
            ),
            dict(datamodule_cfg=dict(max_encoder_length=5, max_prediction_length=3)),
            dict(
                hidden_size=24,
                attention_head_size=8,
                datamodule_cfg=dict(
                    max_encoder_length=5,
                    max_prediction_length=3,
                    add_relative_time_idx=False,
                ),
            ),
            dict(
                hidden_size=12,
                datamodule_cfg=dict(max_encoder_length=7, max_prediction_length=10),
            ),
            dict(attention_head_size=2),
        ]

        default_dm_cfg = {"max_encoder_length": 4, "max_prediction_length": 3}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)

            param["datamodule_cfg"] = default_dm_cfg

        return params
