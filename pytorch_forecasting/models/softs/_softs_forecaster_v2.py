"""
SOFTS forecaster.
"""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class SOFTSForecaster(BaseForecaster):
    """
    SOFTS forecaster (v2).
    Reference : https://arxiv.org/abs/2404.14197

    Parameters
    ----------
    hidden_size: int
        Embedding size of individual time series channel, default = 512
    d_core: int
        Hidden dimension of the central core node, default = 512
    d_ff: int
        Dimension of the feed-forward network, default = 2048
    n_layers: int
        Number of encoder layers, default = 2
    dropout: float
        Dropout rate, default = 0.1
    use_revin: bool
        Whether to use RevIN, default = True
    loss: nn.Module, optional
        Loss function for training. Defaults to ``MAE()``.
    logging_metrics: list[nn.Module] | None
        Metrics to log during training, default = None
    optimizer: Optimizer | str
        Optimizer to use for training, default = "adam"
    optimizer_params: dict | None
        Parameters for the optimizer, default = None
    lr_scheduler: str | None
        Learning rate scheduler to use, default = None
    lr_scheduler_params: dict | None
        Parameters for the learning rate scheduler, default = None
    trainer : lightning.pytorch.Trainer, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    datamodule : EncoderDecoderTimeSeriesDataModule, optional
        Configured, data-less datamodule. See
        :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    ckpt_path : str or Path, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    """

    _tags = {
        "info:name": "SOFTS",
        "info:y_type": ["numeric"],
        "info:compute": 2,
        "authors": ["Secilia-Cxy", "Muhammad-Rebaal"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        hidden_size: int = 512,
        d_core: int = 512,
        d_ff: int = 2048,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_revin: bool = True,
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
        self.d_core = d_core
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_revin = use_revin
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
        from pytorch_forecasting.models.softs._softs_v2 import SOFTS

        return SOFTS

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
            d_core=self.d_core,
            d_ff=self.d_ff,
            n_layers=self.n_layers,
            dropout=self.dropout,
            use_revin=self.use_revin,
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
        list of dict
            Each dict is a valid set of constructor arguments for ``SOFTS``.
            The key ``datamodule_cfg`` is passed to the DataModule, not the model.
        """
        from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE

        params = [
            {},
            dict(hidden_size=64, d_core=64, d_ff=256, n_layers=1),
            dict(hidden_size=128, n_layers=1, use_revin=False),
            dict(
                hidden_size=64,
                n_layers=1,
                loss=MAE(),
            ),
            dict(
                hidden_size=64,
                n_layers=1,
                loss=MAPE(),
            ),
            dict(
                hidden_size=64,
                n_layers=1,
                loss=RMSE(),
            ),
            dict(
                hidden_size=64,
                n_layers=1,
                use_revin=False,
                loss=MAE(),
            ),
            dict(hidden_size=64, dropout=0.0, n_layers=1),
            dict(datamodule_cfg=dict(max_encoder_length=16, max_prediction_length=4)),
            dict(
                optimizer="adamw",
                lr_scheduler="cosine_annealing",
                lr_scheduler_params={"T_max": 5},
            ),
            dict(
                optimizer="adagrad",
                optimizer_params={"lr": 1e-3},
            ),
            dict(hidden_size=64, n_layers=1, logging_metrics=[SMAPE()]),
        ]

        default_dm_cfg = {"max_encoder_length": 8, "max_prediction_length": 2}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            param["datamodule_cfg"] = {**default_dm_cfg, **current_dm_cfg}

        return params
