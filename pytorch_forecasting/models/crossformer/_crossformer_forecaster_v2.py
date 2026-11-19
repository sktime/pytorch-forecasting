"""
CrossFormer forecaster (v2).
"""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class CrossFormerForecaster(BaseForecaster):
    """CrossFormer forecaster (v2).

    Parameters
    ----------
    seg_len: int, default=5
        Length of each segment the input sequence is split into.
    win_size: int, default=2
        Window size for segment merging between encoder scales.
    factor: int, default=10
        Number of routers used in the cross-dimension stage of the TSA layer.
    d_model: int, default=256
        Dimension of the model.
    d_ff: int, optional
        Dimension of the feed-forward network in the TSA layer.
        Defaults to ``4 * d_model``.
    n_heads: int, default=8
        Number of attention heads. ``d_model`` must be divisible by ``n_heads``.
    e_layers: int, default=3
        Number of encoder scale blocks.
    dropout: float, default=0.1
        Dropout rate.
    baseline: bool, default=False
        If True, add the mean of the target history to the predictions.
    loss: nn.Module, optional
        Loss function for training step. Defaults to ``MAE()``.
    logging_metrics: Optional[list[nn.Module]], default=None
        List of metrics to log during training, validation, and testing.
    optimizer: Optional[Union[Optimizer, str]], default='adam'
        Optimizer to use for training.
    optimizer_params: Optional[dict], default=None
        Parameters for the optimizer.
    lr_scheduler: Optional[str], default=None
        Learning rate scheduler to use.
    lr_scheduler_params: Optional[dict], default=None
        Parameters for the learning rate scheduler.
    trainer : lightning.pytorch.Trainer, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    datamodule : TslibDataModule, optional
        Configured, data-less datamodule. See
        :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    ckpt_path : str or Path, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    """

    _tags = {
        "info:name": "CrossFormer",
        "info:compute": 3,
        "authors": ["Yunhao Zhang", "Junchi Yan"],
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        seg_len: int = 5,
        win_size: int = 2,
        factor: int = 10,
        d_model: int = 256,
        d_ff: int | None = None,
        n_heads: int = 8,
        e_layers: int = 3,
        dropout: float = 0.1,
        baseline: bool = False,
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
        self.seg_len = seg_len
        self.win_size = win_size
        self.factor = factor
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.dropout = dropout
        self.baseline = baseline
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
        from pytorch_forecasting.models.crossformer._crossformer_v2 import CrossFormer

        return CrossFormer

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module._tslib_data_module import (
            TslibDataModule,
        )

        return TslibDataModule

    def get_model_params(self) -> dict[str, Any]:
        """Kwargs for ``get_cls()``, with ``None`` sentinels resolved."""
        from pytorch_forecasting.metrics import MAE

        return dict(
            seg_len=self.seg_len,
            win_size=self.win_size,
            factor=self.factor,
            d_model=self.d_model,
            d_ff=self.d_ff,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            baseline=self.baseline,
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

        Parameters
        ----------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
        """

        import torch.nn as nn

        from pytorch_forecasting.metrics import SMAPE

        params = [
            {},
            dict(
                d_model=64,
                n_heads=4,
                e_layers=2,
                seg_len=5,
                datamodule_cfg=dict(context_length=30, prediction_length=10),
            ),
            dict(
                loss=nn.L1Loss(),
                d_model=32,
                n_heads=2,
                e_layers=1,
                seg_len=3,
                datamodule_cfg=dict(context_length=12, prediction_length=3),
            ),
            dict(
                logging_metrics=[SMAPE()],
                d_model=64,
                n_heads=4,
                win_size=2,
                factor=4,
                seg_len=6,
                datamodule_cfg=dict(context_length=24, prediction_length=6),
            ),
            dict(
                optimizer="adamw",
                loss=nn.HuberLoss(),
                lr_scheduler="cosine_annealing",
                lr_scheduler_params={"T_max": 5},
                d_model=32,
                n_heads=2,
                e_layers=1,
                seg_len=4,
                datamodule_cfg=dict(context_length=16, prediction_length=4),
            ),
        ]

        return params
