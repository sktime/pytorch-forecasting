"""xLSTMTime package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class xLSTMTime_pkg(_BasePtForecaster):
    """xLSTMTime package container."""

    _tags = {
        "info:name": "xLSTMTime",
        "info:compute": 3,
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
        "authors": ["muslehal", "phoeenniixx"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import xLSTMTime

        return xLSTMTime

    @classmethod
    def get_base_test_params(cls):
        """
        Return testing parameter settings for the trainer.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
        """

        params = [
            {},
            {"xlstm_type": "mlstm"},
            {"num_layers": 2},
            {"xlstm_type": "slstm", "input_projection_size": 32},
            {
                "xlstm_type": "mlstm",
                "decomposition_kernel": 13,
                "dropout": 0.2,
            },
        ]
        defaults = {"hidden_size": 32, "input_size": 1, "output_size": 1}
        for param in params:
            param.update(defaults)
        return params

    @classmethod
    def _get_test_dataloaders_from(cls, params):
        """
        Get dataloaders from parameters.

        Parameters
        ----------
        params: dict
            Parameters to create dataloaders.
            One of the elements in the list returned by ``get_test_train_params``.

        Returns
        -------
        dataloaders: Dict[str, DataLoader]
            Dict of dataloaders created from the parameters.
            Train, validation, and test dataloaders created from the parameters.
        """
        from pytorch_forecasting.tests._data_scenarios import (
            dataloaders_fixed_window_without_covariates,
        )

        return dataloaders_fixed_window_without_covariates()
