"""Tslib-format D2 datamodule components."""

from pytorch_forecasting.data.data_module.tslib._tslib_data_module import (
    TslibDataModule,
)
from pytorch_forecasting.data.data_module.tslib._tslib_datamodule_pkg import (
    TslibDataModule_pkg,
)

__all__ = [
    "TslibDataModule",
    "TslibDataModule_pkg",
]
