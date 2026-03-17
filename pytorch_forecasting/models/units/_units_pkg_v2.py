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
        """Return testing parameter settings for the trainer.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance. ``create_test_instance`` uses the first (or only) dictionary in
            ``params``.
        """
        params = [
            {},
            {
                "patch_len": 8,
                "stride": 4,
            },
            {
                "d_model": 32,
                "n_heads": 4,
                "patch_len": 8,
                "stride": 4,
            },
            {
                "patch_len": 8,
                "stride": 4,
                "datamodule_cfg": {"context_length": 16, "prediction_length": 4},
            },
        ]

        default_dm_cfg = {"context_length": 12, "prediction_length": 4}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)

            param["datamodule_cfg"] = default_dm_cfg

        return params
