"""NBeats v2 package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class NBEATS_v2_pkg_v2(Base_pkg):
    """NBeats v2 package container."""

    _tags = {
        "info:name": "NBEATS_v2",
        "info:compute": 1,
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
        "authors": ["Palak Bakshi"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.models.nbeats._nbeats_v2 import NBEATS_v2

        return NBEATS_v2

    @classmethod
    def get_datamodule_cls(cls):
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def get_base_test_params(cls):
        params = [
            {},
            {"backcast_loss_ratio": 1.0},
        ]

        for param in params:
            dm_cfg = {"context_length": 4, "prediction_length": 3}
            dm_cfg.update(param.get("datamodule_cfg", {}))
            param["datamodule_cfg"] = dm_cfg

        return params
