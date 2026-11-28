"""TimesNet forecaster."""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class TimesNetForecaster(BaseForecaster):
    """TimesNet Forecaster.

    Parameters
    ----------
    d_model : int, default=64
        Hidden embedding dimension of the model.
    d_ff : int, default=64
        Intermediate channel dimension inside the 2D Inception blocks.
    e_layers : int, default=2
        Number of stacked TimesBlock layers.
    top_k : int, default=5
        Number of top dominant periods selected via FFT in each TimesBlock.
    num_kernels : int, default=6
        Number of parallel 2D convolution kernels in each Inception block.
    dropout : float, default=0.1
        Dropout probability applied in the positional embedding layer.
    loss : nn.Module, optional
        Loss function for training. Defaults to ``MAE()``.
    logging_metrics : list[nn.Module] | None, default=None
        Metrics to log during training, validation and testing.
    optimizer : Optimizer | str | None, default='adam'
        Optimizer used for training.
    optimizer_params : dict | None, default=None
        Parameters passed to the optimizer.
    lr_scheduler : str | None, default=None
        Learning rate scheduler.
    lr_scheduler_params : dict | None, default=None
        Parameters passed to the learning rate scheduler.
    trainer : lightning.pytorch.Trainer, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    datamodule : TslibDataModule, optional
        Configured, data-less datamodule. See
        :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    ckpt_path : str or Path, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.

    References
    ----------
    [1] TimesNet: Temporal 2D-Variation Modeling for General Time Series
        Analysis (https://arxiv.org/abs/2210.02186).
    """

    _tags = {
        "info:name": "TimesNet",
        "info:compute": 2,
        "authors": ["Utkarshhs"],
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        d_model: int = 64,
        d_ff: int = 64,
        e_layers: int = 2,
        top_k: int = 5,
        num_kernels: int = 6,
        dropout: float = 0.1,
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
        self.d_model = d_model
        self.e_layers = e_layers
        self.dropout = dropout
        self.loss = loss
        self.logging_metrics = logging_metrics
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.trainer = trainer
        self.datamodule = datamodule
        self.ckpt_path = ckpt_path
        self.d_ff = d_ff
        self.top_k = top_k
        self.num_kernels = num_kernels
        super().__init__(
            trainer=self.trainer, datamodule=self.datamodule, ckpt_path=self.ckpt_path
        )

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.timesnet._timesnet_v2 import TimesNet

        return TimesNet

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import TslibDataModule

        return TslibDataModule

    def get_model_params(self) -> dict[str, Any]:
        """Kwargs for ``get_cls()``, with ``None`` sentinels resolved."""
        from pytorch_forecasting.metrics import MAE

        return dict(
            top_k=self.top_k,
            num_kernels=self.num_kernels,
            d_ff=self.d_ff,
            d_model=self.d_model,
            e_layers=self.e_layers,
            dropout=self.dropout,
            loss=MAE() if self.loss is None else self.loss,
            logging_metrics=self.logging_metrics,
            optimizer=self.optimizer,
            optimizer_params=self.optimizer_params,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_params=self.lr_scheduler_params,
        )

    @classmethod
    def get_test_train_params(cls):
        """
        Return testing parameter settings for the trainer.

        Returns
        -------
        list[dict]
            Parameter configurations used to create testing instances of the TimesNet
            class.
        """

        import torch.nn as nn

        from pytorch_forecasting.metrics import SMAPE

        params = [
            {},
            dict(
                d_model=16,
                e_layers=1,
                dropout=0.1,
                d_ff=16,
                top_k=2,
                num_kernels=2,
                logging_metrics=[SMAPE()],
            ),
            dict(
                d_model=32,
                e_layers=1,
                dropout=0.0,
                d_ff=16,
                top_k=2,
                num_kernels=2,
                loss=nn.MSELoss(),
            ),
            dict(
                d_model=16,
                e_layers=1,
                dropout=0.1,
                d_ff=16,
                top_k=2,
                num_kernels=2,
                optimizer="adamw",
                optimizer_params={"lr": 1e-3},
            ),
        ]

        default_dm_cfg = {
            "context_length": 8,
            "prediction_length": 2,
        }

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            param["datamodule_cfg"] = {
                **default_dm_cfg,
                **current_dm_cfg,
            }

        return params
