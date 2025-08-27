"""NBeatsKAN package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class NBeatsKAN_pkg(_BasePtForecaster):
    """NBeatsKAN package container."""

    _tags = {
        "info:name": "NBeatsKAN",
        "info:compute": 1,
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
        "authors": ["Sohaib-Ahmed21"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import NBeatsKAN

        return NBeatsKAN

    @classmethod
    def get_base_test_params(cls):
        """Return testing parameter settings for the trainer.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return [
            {"backcast_loss_ratio": 0.0},  # pure forecast loss
            {"backcast_loss_ratio": 1.0},  # equal forecast/backcast
            {
                "stack_types": ["generic"],
                "expansion_coefficient_lengths": [16],
            },
            {
                "num_blocks": [1, 2],
                "num_block_layers": [2, 3],
            },  # varying block structure
            {
                "num": 7,
                "k": 4,
                "sparse_init": True,
                "grid_range": [-0.5, 0.5],
                "sp_trainable": False,
            },  # complex KAN config
        ]

    @classmethod
    def _get_test_dataloaders_from(cls, params):
        """Get dataloaders from parameters."""
        from pytorch_forecasting.tests._data_scenarios import (
            dataloaders_fixed_window_without_covariates,
        )

        return dataloaders_fixed_window_without_covariates()
