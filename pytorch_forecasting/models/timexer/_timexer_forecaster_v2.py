"""
TimeXer forecaster.
"""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class TimeXerForecaster(BaseForecaster):
    """TimeXer forecaster (v2).

    Parameters
    ----------
    enc_in: int, optional
        Number of input features for the encoder. If not provided, it will be set to
        the number of continuous features in the dataset.
    hidden_size: int, default=512
        Dimension of the model embeddings and hidden representations of features.
    n_heads: int, default=8
        Number of attention heads in the multi-head attention mechanism.
    e_layers: int, default=2
        Number of encoder layers in the transformer architecture.
    d_ff: int, default=2048
        Dimension of the feed-forward network in the transformer architecture.
    dropout: float, default=0.1
        Dropout rate for regularization. This is used throughout the model to prevent
        overfitting.
    patch_length: int, default=4
        Length of each non-overlapping patch for endogenous variable tokenization.
    factor: int, default=5
        Factor for the attention mechanism, controlling the number of keys and values.
    activation: str, default='relu'
        Activation function to use in the feed-forward network. Common choices are
        'relu', 'gelu', etc.
    use_efficient_attention: bool, default=False
        If set to True, will use PyTorch's native, optimized Scaled Dot Product
        Attention implementation which can reduce computation time and memory
        consumption for longer sequences. PyTorch automatically selects the
        optimal backend (FlashAttention-2, Memory-Efficient Attention, or their
        own C++ implementation) based on user's input properties, hardware
        capabilities, and build configuration.
    loss: nn.Module, optional
        Loss function to use for training. Defaults to ``MAE()``.
    logging_metrics: Optional[list[nn.Module]], default=None
        List of metrics to log during training, validation, and testing.
    optimizer: Optional[Union[Optimizer, str]], default='adam'
        Optimizer to use for training. Can be a string name or an instance of an
        optimizer.
    optimizer_params: Optional[dict], default=None
        Parameters for the optimizer. If None, default parameters for the optimizer
        will be used.
    lr_scheduler: Optional[str], default=None
        Learning rate scheduler to use. If None, no scheduler is used.
    lr_scheduler_params: Optional[dict], default=None
        Parameters for the learning rate scheduler. If None, default parameters for
        the scheduler will be used.
    trainer : lightning.pytorch.Trainer, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    datamodule : TslibDataModule, optional
        Configured, data-less datamodule. See
        :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    ckpt_path : str or Path, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    """

    _tags = {
        "info:name": "TimeXer",
        "authors": ["PranavBhatP"],
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
        enc_in: int | None = None,
        hidden_size: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        patch_length: int = 4,
        factor: int = 5,
        activation: str = "relu",
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
        self.enc_in = enc_in
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.patch_length = patch_length
        self.factor = factor
        self.activation = activation
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
        from pytorch_forecasting.models.timexer._timexer_v2 import TimeXer

        return TimeXer

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
            enc_in=self.enc_in,
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            patch_length=self.patch_length,
            factor=self.factor,
            activation=self.activation,
            use_efficient_attention=self.use_efficient_attention,
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

        from pytorch_forecasting.metrics import QuantileLoss

        params = [
            {},
            dict(
                hidden_size=64,
                n_heads=4,
            ),
            dict(
                loss=nn.L1Loss(),
                hidden_size=32,
                n_heads=2,
            ),
            dict(datamodule_cfg=dict(context_length=12, prediction_length=3)),
            dict(
                hidden_size=32,
                n_heads=2,
                datamodule_cfg=dict(
                    context_length=12,
                    prediction_length=3,
                    add_relative_time_idx=False,
                ),
            ),
            dict(
                hidden_size=128,
                patch_length=12,
                datamodule_cfg=dict(context_length=16, prediction_length=4),
            ),
            dict(
                n_heads=2,
                e_layers=1,
                patch_length=6,
            ),
            dict(
                hidden_size=256,
                n_heads=8,
                e_layers=3,
                d_ff=1024,
                patch_length=8,
                factor=3,
                activation="gelu",
                dropout=0.2,
            ),
            dict(
                hidden_size=32,
                n_heads=2,
                e_layers=1,
                d_ff=64,
                patch_length=4,
                factor=2,
                activation="relu",
                dropout=0.05,
                datamodule_cfg=dict(
                    context_length=16,
                    prediction_length=4,
                ),
                loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            ),
            dict(
                hidden_size=32,
                patch_length=1,
                n_heads=4,
                e_layers=1,
                d_ff=32,
                dropout=0.1,
                use_efficient_attention=True,
            ),
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
        default_dm_cfg = {"context_length": 12, "prediction_length": 4}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)

            param["datamodule_cfg"] = default_dm_cfg

        return params
