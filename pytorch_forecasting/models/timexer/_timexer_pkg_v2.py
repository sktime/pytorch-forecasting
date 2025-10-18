"""
Metadata container for TimeXer v2.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class TimeXer_pkg_v2(Base_pkg):
    """TimeXer metadata container."""

    _tags = {
        "info:name": "TimeXer",
        "authors": ["PranavBhatP"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.timexer._timexer_v2 import TimeXer

        return TimeXer

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule

        return TslibDataModule

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
                hidden_size=64,
                n_heads=4,
            ),
            dict(datamodule_cfg=dict(context_length=12, prediction_length=3)),
            dict(
                hidden_size=32,
                n_heads=2,
                datamodule_cfg=dict(
                    context_length=12,
                    prediction_length=3,
                    add_relative_time_idx=False,
                ),
            ),
            dict(
                hidden_size=128,
                patch_length=12,
                datamodule_cfg=dict(context_length=16, prediction_length=4),
            ),
            dict(
                n_heads=2,
                e_layers=1,
                patch_length=6,
            ),
            dict(
                hidden_size=256,
                n_heads=8,
                e_layers=3,
                d_ff=1024,
                patch_length=8,
                factor=3,
                activation="gelu",
                dropout=0.2,
            ),
            dict(
                hidden_size=32,
                n_heads=2,
                e_layers=1,
                d_ff=64,
                patch_length=4,
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
