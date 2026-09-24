"""NHiTS v2 package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class NHiTS_pkg_v2(Base_pkg):
    """Package container for the v2 NHiTS implementation."""

    _tags = {
        "info:name": "NHiTS",
        "info:compute": 2,
        "authors": ["mandeepsingh2007"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.nhits._nhits_v2 import NHiTS

        return NHiTS

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def get_test_train_params(cls):
        """Return test configurations used by the generic v2 package tests."""
        params = [
            {},
            dict(hidden_size=16),
            dict(
                hidden_size=32,
                n_blocks=[1, 1],
                n_layers=[2, 2],
                pooling_sizes=[2, 1],
                downsample_frequencies=[2, 1],
            ),
            dict(
                hidden_size=24,
                n_blocks=[1, 1, 1],
                n_layers=[1, 1, 1],
                pooling_mode="average",
            ),
        ]

        default_dm_cfg = {"context_length": 8, "prediction_length": 3}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = default_dm_cfg

        return params
