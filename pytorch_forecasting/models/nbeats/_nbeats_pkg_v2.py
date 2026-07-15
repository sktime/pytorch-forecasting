"""NBeats package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class NBeats_pkg_v2(Base_pkg):
    """NBeats package container."""

    _tags = {
        "info:name": "NBeats",
        "authors": ["jdb78"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
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
        """Return testing parameter settings for the trainer.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from pytorch_forecasting.metrics import QuantileLoss

        params = [
            {},
            dict(
                stack_types=["generic"],
                num_blocks=[1],
                num_block_layers=[4],
                widths=[16],
                backcast_loss_ratio=1.0,
            ),
            dict(
                stack_types=["trend", "seasonality"],
                num_blocks=[2, 2],
                widths=[16, 32],
                backcast_loss_ratio=0.5,
            ),
            dict(
                loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
                stack_types=["generic"],
                num_blocks=[1],
                widths=[16],
            ),
        ]

        default_dm_cfg = {"max_encoder_length": 8, "max_prediction_length": 2}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)

            param["datamodule_cfg"] = default_dm_cfg

        return params
