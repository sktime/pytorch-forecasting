from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class NBEATS_v2_pkg(_BasePtForecaster):

    _tags = {
        "info:name": "NBEATS_v2",
        "info:compute": 1,
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
        "authors": ["jdb78"],
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
    def get_base_test_params(cls):
        return [{}, {"backcast_loss_ratio": 1.0}]
