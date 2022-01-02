"""
Forecasting loggers for timeseries forecasting.
"""

from pytorch_forecasting.loggers.wandb_logger import ForecastingWandbLogger

from pytorch_forecasting.loggers.tensorboard_logger import ForecastingTensorBoardLogger

__all__ = ["ForecastingTensorBoardLogger", "ForecastingWandbLogger"]
