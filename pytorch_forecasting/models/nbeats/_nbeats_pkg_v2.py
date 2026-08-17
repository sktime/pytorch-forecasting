"""NBeats v2 package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class NBeats_pkg_v2(Base_pkg):
    """
    NBeats v2 package container.

    Acts as an orchestrator for the N-BEATS v2 model, providing a streamlined
    configuration-driven interface for dataset handling, model instantiation,
    training, and inference.
    """

    _tags = {
        "info:name": "NBeats_v2",
        "info:compute": 1,
        "authors": ["jdb78", "harshsomankar123-tech"],
        "info:y_type": ["numeric"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """
        Get the underlying model class.

        Returns
        -------
        type
            The ``NBeats_v2`` class.
        """
        from pytorch_forecasting.models.nbeats._nbeats_v2 import NBeats_v2

        return NBeats_v2

    @classmethod
    def get_datamodule_cls(cls):
        """
        Get the underlying DataModule class.

        Returns
        -------
        type
            The ``EncoderDecoderTimeSeriesDataModule`` class.
        """
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
        params : list[dict]
            Parameters to create testing instances of the package.
            Each dictionary contains parameters passed to instantiate
            the package and the underlying model.
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
