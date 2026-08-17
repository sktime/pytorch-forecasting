"""LSTM package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class LSTMModel_pkg(_BasePtForecaster):
    """LSTM package container."""

    _tags = {
        "info:name": "LSTM",
        "info:compute": 1,
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
        "authors": ["Varshith-Yadav"],
        "capability:exogenous": False,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.lstm import LSTMModel

        return LSTMModel

    @classmethod
    def get_base_test_params(cls):
        """Return testing parameter settings."""
        return [
            {},
            {"hidden_size": 8, "n_layers": 1},
        ]

    @classmethod
    def _get_test_dataloaders_from(cls, params):
        """Get dataloaders from parameters."""
        from pytorch_forecasting.tests._data_scenarios import (
            dataloaders_fixed_window_without_covariates,
        )

        return dataloaders_fixed_window_without_covariates()
