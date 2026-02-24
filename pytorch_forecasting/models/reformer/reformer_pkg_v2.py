"""
Metadata container for Reformer v2.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class Reformer_pkg_v2(Base_pkg):
    """Reformer metadata container."""

    _tags = {
        "info:name": "Reformer",
        "authors": ["lucifer4073"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.reformer.reformer_v2 import Reformer

        return Reformer

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
                d_model=64,
                n_heads=4,
            ),
            dict(datamodule_cfg=dict(context_length=12, prediction_length=3)),
            dict(
                d_model=32,
                n_heads=2,
                bucket_size=2,
                n_hashes=2,
                datamodule_cfg=dict(
                    context_length=12,
                    prediction_length=3,
                    add_relative_time_idx=False,
                ),
            ),
            dict(
                d_model=128,
                e_layers=1,
                d_ff=128,
                datamodule_cfg=dict(context_length=16, prediction_length=4),
            ),
            dict(
                n_heads=2,
                e_layers=1,
                bucket_size=4,
            ),
            dict(
                d_model=256,
                n_heads=8,
                e_layers=3,
                d_ff=1024,
                bucket_size=8,
                n_hashes=4,
                activation="gelu",
                dropout=0.2,
            ),
            dict(
                d_model=32,
                n_heads=2,
                e_layers=1,
                d_ff=64,
                bucket_size=2,
                n_hashes=2,
                activation="relu",
                dropout=0.05,
                datamodule_cfg=dict(
                    context_length=16,
                    prediction_length=4,
                ),
            ),
        ]
        default_dm_cfg = {"context_length": 12, "prediction_length": 4}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)

            param["datamodule_cfg"] = default_dm_cfg

        return params
