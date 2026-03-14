"""SCINet v2 package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class SCINet_v2_pkg_v2(Base_pkg):
    """SCINet v2 package container."""

    _tags = {
        "info:name": "SCINet_v2",
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
        SCINet_v2 : type
            The model class.
        """
        from pytorch_forecasting.models.scinet._scinet_v2 import SCINet_v2

        return SCINet_v2

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
            dict(
                datamodule_cfg={
                    "max_encoder_length": 8,
                    "max_prediction_length": 4,
                }
            ),
            dict(
                num_stacks=2,
                num_levels=2,
                hid_size=2,
                datamodule_cfg={
                    "max_encoder_length": 8,
                    "max_prediction_length": 4,
                },
            ),
            dict(
                num_levels=1,
                kernel_size=3,
                dropout=0.1,
                logging_metrics=[SMAPE()],
                datamodule_cfg={
                    "max_encoder_length": 8,
                    "max_prediction_length": 4,
                },
            ),
            dict(
                num_levels=2,
                loss=MAE(),
                datamodule_cfg={
                    "max_encoder_length": 8,
                    "max_prediction_length": 4,
                },
            ),
        ]

        return params
