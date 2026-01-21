"""iTransformer package container v2."""

from pytorch_forecasting.base._base_pkg import Base_pkg

class iTransformer_pkg_v2(Base_pkg):
    """iTransformer metadata container."""

    _tags = {
        "info:name": "iTransformer",
        "authors": ["JATAYU000"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.itransformer._itransformer_v2 import (
            iTransformer,
        )

        return iTransformer

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def get_test_train_params(cls):
        """Get test train params."""
        from pytorch_forecasting.metrics import QuantileLoss

        params =  [
            {},
            dict(d_model=16, n_heads=2, e_layers=2, d_ff=64),
            dict(
                d_model=32,
                n_heads=4,
                e_layers=3,
                d_ff=128,
                dropout=0.1,
                datamodule_cfg=dict(
                    batch_size=4,context_length=12, prediction_length=3)
            ),
            dict(
                hidden_size=32,
                n_heads=2,
                e_layers=1,
                d_ff=64,
                factor=2,
                activation="relu",
                dropout=0.05,
                datamodule_cfg=dict(
                    context_length=16,
                    prediction_length=4,
                ),
                loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            ),
        ]
        default_dm_cfg = {"context_length": 12, "prediction_length": 4}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)

            param["datamodule_cfg"] = default_dm_cfg

        return params
