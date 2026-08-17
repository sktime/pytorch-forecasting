"""LSTMModel package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class LSTMModel_pkg(_BasePtForecaster):
    """LSTMModel package container."""

    _tags = {
        "info:name": "LSTMModel",
        "info:compute": 2,
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
        "authors": ["sktime/pytorch-forecasting contributors"],
        "capability:exogenous": False,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.lstm._lstm import LSTMModel

        return LSTMModel

    @classmethod
    def get_base_test_params(cls):
        """Return testing parameter settings for the trainer."""
        return [
            {"n_layers": 1, "hidden_size": 10},
            {"n_layers": 2, "hidden_size": 16, "dropout": 0.1},
        ]

    @classmethod
    def _get_test_dataloaders_from(cls, params):
        """Get dataloaders from parameters."""
        from pytorch_forecasting.tests._data_scenarios import (
            dataloaders_fixed_window_without_covariates,
        )

        return dataloaders_fixed_window_without_covariates()
