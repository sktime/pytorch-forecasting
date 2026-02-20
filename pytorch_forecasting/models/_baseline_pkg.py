"""Baseline package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class Baseline_pkg(_BasePtForecaster):
    """Baseline package container."""

    _tags = {
        "info:name": "Baseline",
        "info:compute": 0,
        "info:pred_type": ["point"],
        "info:y_type": ["numeric", "category"],
        "authors": ["jdb78"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
        "tests:skip_by_name": [
            "test_integration",
        ],
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import Baseline

        return Baseline

    @classmethod
    def get_base_test_params(cls):
        """Return testing parameter settings for the trainer.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
        """
        return [{}]

    @classmethod
    def _get_test_dataloaders_from(cls, params):
        """Get dataloaders from parameters.

        Parameters
        ----------
        params : dict
            Parameters to create dataloaders.

        Returns
        -------
        dataloaders : dict with keys "train", "val", "test", values torch DataLoader
        """
        from pytorch_forecasting.tests._data_scenarios import (
            dataloaders_fixed_window_without_covariates,
        )

        return dataloaders_fixed_window_without_covariates()
