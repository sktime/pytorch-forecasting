"""
ITransformer forecaster (v2 P-layer).
"""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class ITransformerForecaster(BaseForecaster):
    """iTransformer forecaster (v2).

    High-level ``sklearn``-like wrapper for ``ITransformer``.

    Parameters
    ----------
    d_model : int, default=512
        Dimension of the token embeddings and Transformer hidden size.
    n_heads : int, default=8
        Number of attention heads in the multi-head self-attention blocks.
    e_layers : int, default=3
        Number of Transformer encoder layers.
    d_ff : int, default=512
        Feed-forward hidden dimension inside each encoder layer.
    dropout : float, default=0.1
        Dropout rate used throughout the model.
    activation : str, default='gelu'
        Activation function inside the feed-forward sub-layer.
        One of ``'relu'`` or ``'gelu'``.
    factor : int, default=1
        Attention factor passed to ``FullAttention``.
    use_norm : bool, default=True
        Whether to apply instance normalisation inside the model.
        See ``ITransformer`` docstring for details.
    loss : nn.Module, optional
        Loss function for training step. Defaults to ``MAE()``.
    logging_metrics : list[nn.Module] | None, default=None
        List of metrics to log during training, validation, and testing.
    optimizer : Optimizer | str | None, default='adam'
        Optimizer to use for training.
    optimizer_params : dict | None, default=None
        Parameters for the optimizer.
    lr_scheduler : str | None, default=None
        Learning rate scheduler.
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
        "info:name": "ITransformer",
        "info:compute": 3,
        "authors": ["Dhruv-Kumar-Singh"],
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        factor: int = 1,
        use_norm: bool = True,
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
        self.dropout = dropout
        self.activation = activation
        self.factor = factor
        self.use_norm = use_norm
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
            trainer=self.trainer,
            datamodule=self.datamodule,
            ckpt_path=self.ckpt_path,
        )

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.itransformer._itransformer_v2 import (
            ITransformer,
        )

        return ITransformer

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
            dropout=self.dropout,
            activation=self.activation,
            factor=self.factor,
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
        """Return testing parameter settings for the trainer.

        Returns
        -------
        params : list of dict
            Each dict is a parameter configuration for constructing a test
            instance of ``ITransformerForecaster``.  The first element is
            always ``{}`` to test the class defaults.
        """
        from pytorch_forecasting.metrics import SMAPE, QuantileLoss

        params = [
            {},
            dict(
                d_model=16,
                n_heads=2,
                e_layers=1,
                d_ff=32,
                dropout=0.0,
                activation="gelu",
                datamodule_cfg=dict(context_length=12, prediction_length=4),
            ),
            dict(
                d_model=16,
                n_heads=2,
                e_layers=1,
                d_ff=32,
                use_norm=False,
                logging_metrics=[SMAPE()],
                datamodule_cfg=dict(context_length=8, prediction_length=2),
            ),
            dict(
                d_model=16,
                n_heads=2,
                e_layers=1,
                d_ff=32,
                loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
                datamodule_cfg=dict(context_length=12, prediction_length=4),
            ),
            dict(
                optimizer="adamw",
                lr_scheduler="cosine_annealing",
                lr_scheduler_params={"T_max": 5},
                d_model=16,
                n_heads=2,
                e_layers=1,
                d_ff=32,
            ),
        ]

        default_dm_cfg = {"context_length": 12, "prediction_length": 4}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            param["datamodule_cfg"] = {**default_dm_cfg, **current_dm_cfg}

        return params
