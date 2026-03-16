"""
Packages container for UniTS model.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class UniTS_pkg_v2(Base_pkg):
    """
    UniTS: Unified Time Series Model.
    Reference: https://arxiv.org/abs/2403.00131
    """

    _tags = {
        "info:name": "UniTS",
        "authors": ["Muhammad-Rebaal", "sohamukute"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.models.units._units_v2 import UniTS

        return UniTS

    @classmethod
    def get_datamodule_cls(cls):
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def get_test_train_params(cls):
        """Define varied configurations for auto-testing."""
        return [
            {
                "patch_len": 8,
                "stride": 4,
                "datamodule_cfg": {"context_length": 12, "prediction_length": 4},
            },
            {
                "d_model": 32,
                "n_heads": 4,
                "patch_len": 8,
                "stride": 4,
                "datamodule_cfg": {"context_length": 12, "prediction_length": 4},
            },
            {
                "patch_len": 8,
                "stride": 4,
                "datamodule_cfg": {"context_length": 16, "prediction_length": 4},
            },
        ]
