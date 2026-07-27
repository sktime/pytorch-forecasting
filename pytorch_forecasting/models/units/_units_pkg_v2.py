"""
Packages container for UniTS model.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class UniTS_pkg_v2(Base_pkg):
    """
    UniTS: Unified Time Series Model.
    Reference: https://arxiv.org/abs/2403.00131
    Github: https://github.com/mims-harvard/UniTS
    """

    _tags = {
        "info:name": "UniTS",
        "info:compute": 4,
        "authors": ["Muhammad-Rebaal", "gasvn", "sohamukute"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "info:pred_type": ["point", "quantile", "distribution"],
        "info:y_type": ["numeric"],
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.models.units._units_v2 import UniTS

        return UniTS

    @classmethod
    def get_datamodule_cls(cls):
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    @classmethod
    def get_base_test_params(cls):
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
        from pytorch_forecasting.metrics import NormalDistributionLoss, QuantileLoss

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
                "datamodule_cfg": {
                    "max_encoder_length": 16,
                    "max_prediction_length": 4,
                },
            },
            {
                "patch_len": 8,
                "stride": 4,
                "loss": QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            },
            {
                "patch_len": 8,
                "stride": 4,
                "loss": NormalDistributionLoss(),
            },
        ]

        base_dm_cfg = {"max_encoder_length": 16, "max_prediction_length": 4}

        for param in params:
            merged = base_dm_cfg.copy()
            merged.update(param.get("datamodule_cfg", {}))
            param["datamodule_cfg"] = merged

        return params
