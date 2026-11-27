"""PAttn forecaster."""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class PAttnForecaster(BaseForecaster):
    """PAttn Forecaster.

    Parameters
    ----------
    d_model : int, default=64
        Hidden dimension of the patch embeddings and attention layers.
    n_heads : int, default=8
        Number of attention heads. Must divide ``d_model``.
    e_layers : int, default=1
        Number of stacked self-attention encoder layers.
    d_ff : int, default=256
        Hidden dimension of the feed-forward network inside each encoder layer.
    patch_len : int, default=16
        Length of each time series patch.
    stride : int, default=8
        Stride between consecutive patches.
    dropout : float, default=0.1
        Dropout probability used in attention and encoder layers.
    factor : int, default=5
        Attention factor passed to ``FullAttention``.
    activation : str, default="gelu"
        Activation function used in the encoder feed-forward network
        (``"relu"`` or ``"gelu"``).
    loss : nn.Module, optional
        Loss function for training. Defaults to ``MAE()``.
    logging_metrics : list[nn.Module] | None, default=None
        Metrics to log during training, validation, and testing.
    optimizer : Optimizer | str | None, default="adam"
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
    [1] Are Language Models Actually Useful for Time Series Forecasting?
        (https://arxiv.org/abs/2406.16964).
    """

    _tags = {
        "info:name": "PAttn",
        "info:compute": 2,
        "authors": ["tuhinbuilds"],
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
        n_heads: int = 8,
        e_layers: int = 1,
        d_ff: int = 256,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        factor: int = 5,
        activation: str = "gelu",
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
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.patch_len = patch_len
        self.stride = stride
        self.dropout = dropout
        self.factor = factor
        self.activation = activation
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
        from pytorch_forecasting.models.pattn._pattn_v2 import PAttn

        return PAttn

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import TslibDataModule

        return TslibDataModule

    def get_model_params(self) -> dict[str, Any]:
        """Kwargs for ``get_cls()``, with ``None`` sentinels resolved."""
        from pytorch_forecasting.metrics import MAE

        return dict(
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            patch_len=self.patch_len,
            stride=self.stride,
            dropout=self.dropout,
            factor=self.factor,
            activation=self.activation,
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
            Parameter configurations used to create testing instances of the
            ``PAttnForecaster`` class.
        """
        import torch.nn as nn

        from pytorch_forecasting.metrics import SMAPE, QuantileLoss

        params = [
            {},
            dict(
                d_model=32,
                n_heads=4,
                e_layers=2,
                d_ff=64,
                patch_len=4,
                stride=2,
                dropout=0.1,
                logging_metrics=[SMAPE()],
                datamodule_cfg={"context_length": 12, "prediction_length": 2},
            ),
            dict(
                d_model=16,
                n_heads=2,
                d_ff=32,
                patch_len=4,
                stride=2,
                dropout=0.0,
                activation="relu",
                loss=nn.MSELoss(),
            ),
            dict(
                d_model=16,
                n_heads=2,
                d_ff=32,
                patch_len=4,
                stride=2,
                loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            ),
            dict(
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
