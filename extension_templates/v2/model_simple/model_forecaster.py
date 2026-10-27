"""Template for the forecaster that configures and runs a v2 model."""

from pathlib import Path
from typing import Any

from lightning import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import Metric
from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class MyModelForecaster(BaseForecaster):
    """Configure a model and its compatible data module for forecasting."""

    _tags = {
        "info:name": "MyModel",
        "info:compute": 2,
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
        "authors": ["your-github-handle"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
        "python_dependencies": [],
    }

    def __init__(
        self,
        parama: Any = None,
        paramb: str = "default",
        paramc: Any = None,
        loss: Metric | nn.Module | None = None,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        trainer: Trainer | None = None,
        datamodule: Any = None,
        ckpt_path: str | Path | None = None,
    ):
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc
        self.loss = loss
        self.logging_metrics = logging_metrics
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        super().__init__(trainer=trainer, datamodule=datamodule, ckpt_path=ckpt_path)

    @classmethod
    def get_cls(cls):
        """Return the underlying model class."""
        from pytorch_forecasting.models.my_model.model import MyModel

        return MyModel

    @classmethod
    def get_datamodule_cls(cls):
        """Return the concrete data module compatible with the model."""
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    def get_model_params(self) -> dict[str, Any]:
        """Return model constructor arguments, excluding data-derived metadata."""
        from pytorch_forecasting.metrics import MAE

        return {
            "parama": self.parama,
            "paramb": self.paramb,
            "paramc": self.paramc,
            "loss": MAE() if self.loss is None else self.loss,
            "logging_metrics": self.logging_metrics,
            "optimizer": self.optimizer,
            "optimizer_params": self.optimizer_params,
            "lr_scheduler": self.lr_scheduler,
            "lr_scheduler_params": self.lr_scheduler_params,
        }

    @classmethod
    def get_test_train_params(cls) -> list[dict[str, Any]]:
        """Return lightweight forecaster configurations for the v2 test suite."""
        return [{}, {"paramb": "alternate"}]

    # TODO: add model-specific configuration and test parameters.
