"""
``TFTForecaster`` - prototype for the v2 redesign.
"""

from typing import Any

from lightning import Trainer
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting._proto._base_forecaster import BaseForecaster


class TFTForecaster(BaseForecaster):
    """Temporal Fusion Transformer forecaster.

    Parameters
    ----------
    hidden_size : int, default=64
        Size of the hidden layers.
    num_layers : int, default=2
        Number of LSTM layers in encoder and decoder.
    attention_head_size : int, default=4
        Number of attention heads. Must divide ``hidden_size``.
    dropout : float, default=0.1
        Dropout rate.
    output_size : int, default=1
        Number of outputs per time step.
    loss : nn.Module or pytorch-forecasting Metric, optional, default=None
        Loss to train with. Defaults to
        :py:class:`~pytorch_forecasting.metrics.point.MAE`.
    logging_metrics : list of nn.Module, optional, default=None
        Metrics to log during training, validation and testing.
    optimizer : str, Optimizer, or callable, default="adam"
        Optimizer to train with.
    optimizer_params : dict, optional, default=None
        Parameters for the optimizer.
    lr_scheduler : str, optional, default=None
        Learning rate scheduler. One of ``"reduce_lr_on_plateau"``,
        ``"step_lr"``, ``"cosine_annealing"``,
        ``"cosine_annealing_warm_restarts"``.
    lr_scheduler_params : dict, optional, default=None
        Parameters for the learning rate scheduler.
    trainer : lightning.Trainer, optional, default=None
        Trainer to fit with. May also be given to ``fit``.
    datamodule : LightningDataModule, optional, default=None
        Data module to use, configured but without data. Defaults to an
        :py:class:`~pytorch_forecasting.data.data_module.EncoderDecoderTimeSeriesDataModule`
        with default settings.
    """

    _tags = {
        "info:name": "TFT",
        "authors": ["phoeenniixx"],
        "info:compute": 3,
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        output_size: int = 1,
        loss: nn.Module | None = None,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        trainer: Trainer | None = None,
        datamodule: Any = None,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.output_size = output_size
        self.loss = loss
        self.logging_metrics = logging_metrics
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params

        super().__init__(trainer=trainer, datamodule=datamodule)

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import TFT

        return TFT

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    def get_model_params(self) -> dict[str, Any]:
        """Return the parameters to construct the model with."""
        from pytorch_forecasting.metrics import MAE

        return dict(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            output_size=self.output_size,
            loss=MAE() if self.loss is None else self.loss,
            logging_metrics=self.logging_metrics,
            optimizer=self.optimizer,
            optimizer_params=self.optimizer_params,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_params=self.lr_scheduler_params,
        )
