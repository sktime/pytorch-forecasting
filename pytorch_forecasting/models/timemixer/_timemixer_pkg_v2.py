"""
Metadata container for TimeMixer v2.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg

class TimeMixer_pkg_v2 (Base_pkg):
    """TimeMixer metadata container."""

    _tags = {
        "info:name": "TimeMixer",
        "authors": ["lucifer4073"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.timemixer._timemixer_v2 import TimeMixer

        return TimeMixer
    
    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule

        return TslibDataModule
    
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

        pass