"""
Packages container for DeepAR model.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class DeepAR_pkg_v2(Base_pkg):
    """DeepAR package container."""

    _tags = {
        "info:name": "DeepAR",
        "info:compute": 3,
        "info:pred_type": ["distr"],
        "info:y_type": ["numeric"],
        "authors": ["jdb78"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
        "tests:skip_by_name": [
            "test_integration",
            "test_checkpointing",
            "test_predict_modes",
        ],
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
    def get_base_test_params(cls):
        """Return testing parameter settings for the trainer."""
        return [
            {},
            dict(
                cell_type="GRU",
                hidden_size=16,
                rnn_layers=2,
            ),
        ]

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
        from pytorch_forecasting.metrics import (
            LogNormalDistributionLoss,
            NegativeBinomialDistributionLoss,
            NormalDistributionLoss,
        )

        params = [
            {},
            dict(
                cell_type="GRU",
                hidden_size=16,
                rnn_layers=2,
            ),
            dict(
                hidden_size=32,
                rnn_layers=3,
                dropout=0.2,
            ),
            dict(
                loss=NegativeBinomialDistributionLoss(),
                hidden_size=20,
                rnn_layers=2,
            ),
            dict(
                loss=LogNormalDistributionLoss(),
                hidden_size=24,
                rnn_layers=2,
                dropout=0.15,
            ),
            dict(
                hidden_size=20,
                datamodule_cfg=dict(
                    max_encoder_length=7,
                    max_prediction_length=5,
                ),
            ),
            dict(
                hidden_size=16,
                n_validation_samples=50,
                n_plotting_samples=25,
            ),
            dict(
                hidden_size=10,
                rnn_layers=1,
                dropout=0.0,
                datamodule_cfg=dict(
                    max_encoder_length=3,
                    max_prediction_length=2,
                ),
            ),
        ]

        default_dm_cfg = {
            "max_encoder_length": 4,
            "max_prediction_length": 3,
        }

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            merged_dm_cfg = default_dm_cfg.copy()
            merged_dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = merged_dm_cfg

        return params
