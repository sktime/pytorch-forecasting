"""
Template for implementing a model's package container in v2 architecture.
"""

from typing import Any

from pytorch_forecasting.base._base_pkg import Base_pkg


class MyNewModel_pkg(Base_pkg):
    """
    Package container for model registration and testing.

    This class is the entry point for metadata about the model.
    It links the model to its DataModule and provides test configurations.
    """

    # _tags provide metadata about the model's capabilities and information.
    # These are used by the library to determine compatibility and for logging.
    _tags = {
        "info:name": "MyNewModel",
        "info:compute": 3,  # Computational cost (1-5).
        "info:pred_type": ["point"],  # Supported prediction types.
        "info:y_type": ["numeric"],  # Supported target types.
        "authors": ["your_github_username"],  # List of maintainers.
        "capability:exogenous": True,  # Supports exogenous variables.
        "capability:multivariate": True,  # Supports multivariate targets.
        "capability:pred_int": True,  # Supports prediction intervals.
        "capability:flexible_history_length": True,  # Supports variable context.
        "capability:cold_start": False,  # Supports zero-context prediction.
    }

    @classmethod
    def get_cls(cls):
        """
        Get the model class associated with this package.

        Returns:
            type[BaseModel]: The model class.
        """
        from pytorch_forecasting.models.model_template import MyNewModel

        return MyNewModel

    @classmethod
    def get_datamodule_cls(cls):
        """
        Get the standard DataModule class for this model.

        Returns:
            type[BaseDataModule]: The DataModule class.
        """
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    @classmethod
    def get_test_train_params(cls) -> list[dict[str, Any]]:
        """
        Return parameter configurations for automated testing.

        Returns:
        --------
        list[dict[str, Any]]
            Configurations for instantiating the model during tests.
        """
        import torch

        from pytorch_forecasting.metrics import MAE

        # RATIONALE: These are used to test your model across scenarios.
        return [
            dict(hidden_size=16, loss=MAE()),
            dict(hidden_size=32, loss=torch.nn.MSELoss()),
        ]

    @classmethod
    def get_base_test_params(cls) -> list[dict[str, Any]]:
        """
        Optional: Return basic parameters for testing the model.
        """
        return [{"hidden_size": 5}]
