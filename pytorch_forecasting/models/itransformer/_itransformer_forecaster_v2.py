"""
iTransformer forecaster.
"""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class iTransformerForecaster(BaseForecaster):
    """iTransformer forecaster (v2).

    Parameters
    ----------
    hidden_size : int, default=512
        Dimension of the model embeddings.
    n_heads : int, default=8
        Number of attention heads in multi-head self-attention.
    e_layers : int, default=2
        Number of Transformer encoder layers.
    d_ff : int, default=2048
        Dimension of the feed-forward network in each encoder layer.
    dropout : float, default=0.1
        Dropout rate.
    factor : int, default=1
        Attention factor.
    activation : str, default="gelu"
        Activation function (``"relu"`` or ``"gelu"``).
    use_norm : bool, default=True
        Whether to apply non-stationary normalization.
    loss : nn.Module, optional
        Loss function for training step. Defaults to ``MAE()``.
    logging_metrics : list[nn.Module] | None, default=None
        List of metrics to log during training, validation, and testing.
    optimizer : Optimizer | str | None, default="adam"
        Optimizer to use for training.
    optimizer_params : dict | None, default=None
        Parameters for the optimizer.
    lr_scheduler : str | None, default=None
        Learning rate scheduler to use.
    lr_scheduler_params : dict | None, default=None
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
        "info:name": "iTransformer",
        "info:compute": 2,
        "authors": ["Ayushagrawal-cse"],
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        hidden_size: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        factor: int = 1,
        activation: str = "gelu",
        use_norm: bool = True,
        use_efficient_attention: bool = False,
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
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.factor = factor
        self.activation = activation
        self.use_norm = use_norm
        self.use_efficient_attention = use_efficient_attention
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
        from pytorch_forecasting.models.itransformer._itransformer_v2 import (
            iTransformer,
        )

        return iTransformer

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import TslibDataModule

        return TslibDataModule

    def get_model_params(self) -> dict[str, Any]:
        """Kwargs for ``get_cls()``, with ``None`` sentinels resolved."""
        from pytorch_forecasting.metrics import MAE

        return dict(
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            factor=self.factor,
            activation=self.activation,
            use_norm=self.use_norm,
            loss=MAE() if self.loss is None else self.loss,
            logging_metrics=self.logging_metrics,
            optimizer=self.optimizer,
            optimizer_params=self.optimizer_params,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_params=self.lr_scheduler_params,
        )

    @classmethod
    def get_test_train_params(cls):
        """Return testing parameter settings for the trainer."""
        import torch.nn as nn

        from pytorch_forecasting.metrics import SMAPE

        params = [
            dict(hidden_size=16, n_heads=2, e_layers=1, d_ff=32),
            dict(
                hidden_size=16,
                n_heads=2,
                e_layers=2,
                d_ff=32,
                activation="relu",
                use_norm=False,
                logging_metrics=[SMAPE()],
            ),
            dict(
                hidden_size=16,
                n_heads=2,
                e_layers=1,
                d_ff=32,
                loss=nn.MSELoss(),
                optimizer="adamw",
                lr_scheduler="cosine_annealing",
                lr_scheduler_params={"T_max": 5},
            ),
        ]

        default_dm_cfg = {"context_length": 8, "prediction_length": 2}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            dm_cfg = default_dm_cfg.copy()
            dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = dm_cfg

        return params