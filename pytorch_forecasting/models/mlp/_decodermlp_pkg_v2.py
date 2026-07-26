"""DecoderMLP v2 package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class DecoderMLP_pkg_v2(Base_pkg):
    """DecoderMLP v2 package container."""

    _tags = {
        "info:name": "DecoderMLP_v2",
        "info:compute": 1,
        "authors": ["echo-xiao"],
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "capability:multivariate": False,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": True,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.mlp._decodermlp_v2 import DecoderMLP_v2

        return DecoderMLP_v2

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
            Parameters to create testing instances of the class. Each dict is passed
            as ``model_cfg`` to the package constructor; the ``"datamodule_cfg"`` key
            is forwarded to the datamodule constructor.
        """
        from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, QuantileLoss

        params = [
            {},
            dict(
                hidden_size=64, n_hidden_layers=2, dropout=0.1, norm=True, loss=RMSE()
            ),
            dict(
                hidden_size=128,
                n_hidden_layers=1,
                activation_class="ReLU",
                loss=SMAPE(),
                logging_metrics=[MAE()],
            ),
            dict(hidden_size=32, n_hidden_layers=2, norm=False, loss=MAE()),
            dict(hidden_size=64, n_hidden_layers=1, loss=QuantileLoss()),
            dict(
                optimizer="adamw",
                lr_scheduler="cosine_annealing",
                lr_scheduler_params={"T_max": 5},
                loss=MAE(),
            ),
        ]

        default_dm_cfg = {"max_encoder_length": 4, "max_prediction_length": 3}
        for param in params:
            dm_cfg = default_dm_cfg.copy()
            dm_cfg.update(param.get("datamodule_cfg", {}))
            param["datamodule_cfg"] = dm_cfg

        return params
