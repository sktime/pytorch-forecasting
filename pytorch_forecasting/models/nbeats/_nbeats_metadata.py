"""NBeats metadata container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class NBeatsMetadata(_BasePtForecaster):
    """NBeats metadata container."""

    _tags = {
        "info:name": "NBeats",
        "info:compute": 1,
        "authors": ["jdb78"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import NBeats

        return NBeats

    @classmethod
    def get_test_train_params(cls):
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
            {
                "backcast_loss_ratio": 1.0,
                "add_relative_time_idx": True,
            }
        ]
