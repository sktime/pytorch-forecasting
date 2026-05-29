"""
Package container template for a custom neural network (v1).

This class exposes metadata (tags) and links to the actual model class.
Copy and modify this when adding a new model to pytorch-forecasting.
"""

from pytorch_forecasting.models.base import _BasePtForecaster


class ExampleNetwork_pkg(_BasePtForecaster):
    """
    Package container for ExampleNetwork.

    This is required for CI discovery, registration, and testing of v1 models.
    """

    _tags = {
        # Human-readable model name (MUST match the model class)
        "info:name": "ExampleNetwork",
        # What type of predictions this model produces:
        # - ["point"]        → deterministic forecasts
        # - ["quantile"]     → probabilistic quantiles
        # - ["distribution"] → full predictive distribution
        "info:pred_type": ["point"],
        # What type of target the model supports:
        # Typically: ["numeric"]
        "info:y_type": ["numeric"],
        # Whether the model can use exogenous covariates (X)
        # True = uses X in a meaningful way
        # False = ignores X
        "capability:exogenous": True,
        # Whether the model supports multiple target variables
        # True = multivariate forecasting supported
        # False = univariate only
        "capability:multivariate": True,
        # Whether the model supports probabilistic prediction intervals
        "capability:pred_int": False,
        # Whether the model can work with variable-length history
        "capability:flexible_history_length": True,
        # Whether the model can make predictions without long history
        "capability:cold_start": False,
        # Approximate compute cost of the model
        # 1 = lightweight, 5 = very heavy
        "info:compute": 2,
        # GitHub usernames of contributors
        "authors": ["your-github-handle"],
    }

    @classmethod
    def get_cls(cls):
        """Return the actual Lightning model class."""
        from .model import ExampleNetwork

        return ExampleNetwork

    @classmethod
    def get_base_test_params(cls):
        """
        Return minimal test parameters for CI.

        This must return parameters that construct valid model instances.
        Multiple parameter sets are recommended to cover edge cases.
        """
        return [
            {"hidden_size": 8},
            {"hidden_size": 16},
        ]

    @classmethod
    def _get_test_dataloaders_from(cls, dataset, **kwargs):
        """
        Return train and validation dataloaders for testing.

        This is REQUIRED for v1 models in CI.
        """
        train, val = dataset.to_dataloaders(batch_size=16)
        return train, val
