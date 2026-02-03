"""
Package container template for a custom neural network (v1).

This class exposes metadata (tags) and links to the actual model class.
Copy and modify this when adding a new model to pytorch-forecasting.
"""

from pytorch_forecasting.models.base import _BasePtForecaster


class CustomNetwork_pkg(_BasePtForecaster):
    """
    Minimal package container template for a new neural network.

    This is REQUIRED for CI discovery and testing.
    """

    _tags = {
        "info:name": "CustomNetwork",
        # adjust these tags for your actual model
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "authors": ["<your-name>"],
        "info:compute": 2,
    }

    @classmethod
    def get_cls(cls):
        """REQUIRED: return the actual model class."""
        from .model import ExampleNetwork

        return ExampleNetwork

    @classmethod
    def get_base_test_params(cls):
        """
        OPTIONAL: define minimal test parameters.

        Most models can simply return [{}] to use default test settings.
        Add custom parameters here only if your model needs special test setup.
        """
        return [{}]
