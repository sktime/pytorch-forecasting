"""DeepAR v2 package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class DeepAR_pkg_v2(Base_pkg):
    """DeepAR v2 package container.

    Wraps the v2 DeepAR model with the ``EncoderDecoderTimeSeriesDataModule``
    data pipeline. Provides high-level ``fit`` and ``predict`` API through
    ``Base_pkg``.
    """

    _tags = {
        "info:name": "DeepAR",
        "info:compute": 3,
        "authors": ["jdb78", "harshsomankar123-tech"],
        "capability:exogenous": True,
        "capability:multivariate": False,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.deepar._deepar_v2 import DeepAR

        return DeepAR

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
        params : list of dict
            Parameters to create testing instances of the class.
            Each dict contains model config and optionally datamodule_cfg.
        """
        from pytorch_forecasting.metrics import (
            LogNormalDistributionLoss,
            NormalDistributionLoss,
        )

        params = [
            dict(loss=NormalDistributionLoss()),
            dict(loss=NormalDistributionLoss(), cell_type="GRU", dropout=0.2),
            dict(loss=LogNormalDistributionLoss(), optimizer="adamw"),
            dict(
                loss=NormalDistributionLoss(),
                hidden_size=16,
                rnn_layers=1,
                optimizer="sgd",
                optimizer_params=dict(momentum=0.9),
            ),
        ]

        default_dm_cfg = {
            "max_encoder_length": 4,
            "max_prediction_length": 3,
        }

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            merged = {**default_dm_cfg, **current_dm_cfg}
            param["datamodule_cfg"] = merged

        return params
