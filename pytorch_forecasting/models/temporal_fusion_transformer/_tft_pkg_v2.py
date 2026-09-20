"""TFT forecaster: user-facing estimator for the v2 Temporal Fusion Transformer."""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class TFTForecaster(BaseForecaster):
    """Temporal Fusion Transformer forecaster (v2).

    Parameters
    ----------
    hidden_size : int, default=64
        Size of the hidden layers.
    num_layers : int, default=2
        Number of LSTM layers in encoder and decoder.
    attention_head_size : int, default=4
        Number of attention heads.
    dropout : float, default=0.1
        Dropout rate.
    output_size : int, default=1
        Number of outputs per time step, e.g. number of quantiles.
    loss : nn.Module, optional
        Loss to optimise. Defaults to ``MAE()``.
    logging_metrics : list of nn.Module, optional
        Metrics to log during training.
    optimizer : Optimizer or str, default="adam"
        Optimizer, or its name.
    optimizer_params : dict, optional
        Keyword arguments for the optimizer.
    lr_scheduler : str, optional
        Learning-rate scheduler name.
    lr_scheduler_params : dict, optional
        Keyword arguments for the scheduler.
    trainer : lightning.pytorch.Trainer, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    datamodule : EncoderDecoderTimeSeriesDataModule, optional
        Configured, data-less datamodule. See
        :class:`~pytorch_forecasting.models.base.BaseForecaster`.
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
        ckpt_path: str | Path | None = None,
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
        self.trainer = trainer
        self.datamodule = datamodule
        self.ckpt_path = ckpt_path
        super().__init__(
            trainer=self.trainer, datamodule=self.datamodule, ckpt_path=self.ckpt_path
        )

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
        """Kwargs for ``get_cls()``, with ``None`` sentinels resolved."""
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
        import torch.nn as nn

        params = [
            {},
            dict(
                hidden_size=25,
                attention_head_size=5,
            ),
            dict(
                loss=nn.MSELoss(),
                hidden_size=16,
                attention_head_size=4,
            ),
            dict(
                loss=nn.GaussianNLLLoss(),
                output_size=2,
                hidden_size=16,
                attention_head_size=2,
            ),
            dict(datamodule_cfg=dict(max_encoder_length=5, max_prediction_length=3)),
            dict(
                hidden_size=24,
                attention_head_size=8,
                datamodule_cfg=dict(
                    max_encoder_length=5,
                    max_prediction_length=3,
                    add_relative_time_idx=False,
                ),
            ),
            dict(
                hidden_size=12,
                datamodule_cfg=dict(max_encoder_length=7, max_prediction_length=10),
            ),
            dict(attention_head_size=2),
            dict(
                optimizer="adamw",
                lr_scheduler="cosine_annealing",
                lr_scheduler_params={"T_max": 5},
            ),
            dict(
                optimizer="adagrad",
                optimizer_params={"lr": 1e-3},
            ),
        ]

        default_dm_cfg = {"max_encoder_length": 4, "max_prediction_length": 3}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)

            param["datamodule_cfg"] = default_dm_cfg

        return params
