"""UniTS forecaster."""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class UniTSForecaster(BaseForecaster):
    """UniTS Forecaster.

    Parameters
    ----------
    d_model : int, default=64
        Transformer model dimension.
    n_heads : int, default=8
        Number of self-attention heads. Must divide ``d_model``.
    e_layers : int, default=3
        Number of transformer encoder layers.
    d_ff : int, default=512
        Feed-forward hidden dimension.
    dropout : float, default=0.1
        Dropout probability.
    patch_len : int, default=16
        Patch length in time steps. Must not exceed the encoder length.
    stride : int, default=8
        Stride between patches.
    prompt_len : int, default=10
        Number of learnable task-prompt tokens prepended to the patch sequence.
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
    datamodule : EncoderDecoderTimeSeriesDataModule, optional
        Configured, data-less datamodule. See
        :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    ckpt_path : str or Path, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.

    References
    ----------
    [1] UniTS: A Unified Multi-Task Time Series Model
        (https://arxiv.org/abs/2403.00131).
    """

    _tags = {
        "info:name": "UniTS",
        "info:compute": 4,
        "authors": ["Muhammad-Rebaal", "gasvn", "sohamukute"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        patch_len: int = 16,
        stride: int = 8,
        prompt_len: int = 10,
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
        self.patch_len = patch_len
        self.stride = stride
        self.prompt_len = prompt_len
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
        from pytorch_forecasting.models.units._units_v2 import UniTS

        return UniTS

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
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            patch_len=self.patch_len,
            stride=self.stride,
            prompt_len=self.prompt_len,
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
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance. ``create_test_instance`` uses the first (or only) dictionary in
            ``params``.
        """
        from pytorch_forecasting.metrics import QuantileLoss

        params = [
            {},
            {
                "patch_len": 8,
                "stride": 4,
            },
            {
                "d_model": 32,
                "n_heads": 4,
                "patch_len": 8,
                "stride": 4,
            },
            {
                "patch_len": 8,
                "stride": 4,
                "datamodule_cfg": {
                    "max_encoder_length": 16,
                    "max_prediction_length": 4,
                },
            },
            {
                "patch_len": 8,
                "stride": 4,
                "loss": QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            },
        ]

        base_dm_cfg = {"max_encoder_length": 16, "max_prediction_length": 4}

        for param in params:
            merged = base_dm_cfg.copy()
            merged.update(param.get("datamodule_cfg", {}))
            param["datamodule_cfg"] = merged

        return params
