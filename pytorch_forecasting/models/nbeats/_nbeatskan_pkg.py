"""NBeatsKAN package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class NBeatsKAN_pkg(_BasePtForecaster):
    """NBeatsKAN package container."""

    _tags = {
        "info:name": "NBeatsKAN",
        "info:compute": 1,
        "authors": ["Sohaib-Ahmed21"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import NBeatsKAN

        return NBeatsKAN

    @classmethod
    def get_test_train_params(cls):
        """Return testing parameter settings for the trainer."""
        return [{"backcast_loss_ratio": 1.0}]

    @classmethod
    def _get_test_dataloaders_from(cls, params):
        """Get dataloaders from parameters."""
        from pytorch_forecasting.tests._data_scenarios import (
            dataloaders_fixed_window_without_covariates,
        )

        return dataloaders_fixed_window_without_covariates()
