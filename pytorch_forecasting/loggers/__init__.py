"""
Forecasting loggers for timeseries forecasting.
"""

from pytorch_forecasting.loggers.tensorboard_logger import ForecastingTensorBoardLogger
from pytorch_forecasting.loggers.wandb_logger import ForecastingWandbLogger

__all__ = ["ForecastingTensorBoardLogger", "ForecastingWandbLogger"]
