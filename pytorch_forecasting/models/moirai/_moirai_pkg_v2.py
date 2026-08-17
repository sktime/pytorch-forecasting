"""
Moirai package container.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class Moirai_pkg_v2(Base_pkg):
    """Moirai package container."""

    _tags = {
        "info:name": "Moirai",
        "authors": ["SalesforceAIResearch", "ubermensch19"],
        "info:compute": 2,
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "capability:multivariate": False,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.moirai._moirai_v2 import Moirai

        return Moirai

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    @classmethod
    def get_test_train_params(cls):
        return {}
