"""ModernTCN forecaster."""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class ModernTCNForecaster(BaseForecaster):
    """ModernTCN Forecaster.

    Parameters
    ----------
    d_model : int, default=64
        Embedding dimension per patch.
    kernel_size : int, default=51
        Large kernel size for the reparameterizable depthwise convolution.
    small_kernel_size : int, default=5
        Small kernel size for the parallel convolution branch.
    n_blocks : int, default=2
        Number of ModernTCN encoder blocks.
    d_ff : int, default=256
        Hidden dimension in the pointwise convolution FFNs.
    patch_size : int, default=8
        Number of time steps per patch.
    dropout : float, default=0.1
        Dropout rate inside each encoder block.
    head_dropout : float, default=0.1
        Dropout rate in the output head.
    individual : bool, default=False
        If True, use a separate output projection per variable.
    use_revin : bool, default=True
        Whether to apply Reversible Instance Normalization before the encoder.
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
    [1] ModernTCN: A Modern Pure Convolution Structure for General Time Series
        Analysis (https://arxiv.org/abs/2401.17593).
    """

    _tags = {
        "info:name": "ModernTCN",
        "authors": ["Muhammad-Rebaal", "luodhhh"],
        "info:compute": 2,
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
        kernel_size: int = 51,
        small_kernel_size: int = 5,
        n_blocks: int = 2,
        d_ff: int = 256,
        patch_size: int = 8,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
        individual: bool = False,
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
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.small_kernel_size = small_kernel_size
        self.n_blocks = n_blocks
        self.d_ff = d_ff
        self.patch_size = patch_size
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.individual = individual
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
        from pytorch_forecasting.models.modern_tcn._modern_tcn_v2 import ModernTCN

        return ModernTCN

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
            kernel_size=self.kernel_size,
            small_kernel_size=self.small_kernel_size,
            n_blocks=self.n_blocks,
            d_ff=self.d_ff,
            patch_size=self.patch_size,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
            individual=self.individual,
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
        """Return testing parameter settings."""
        from pytorch_forecasting.metrics import QuantileLoss

        params = [
            {},
            {
                "d_model": 16,
                "kernel_size": 3,
                "n_blocks": 1,
                "d_ff": 32,
                "patch_size": 4,
                "use_revin": False,
            },
            {
                "d_model": 8,
                "kernel_size": 3,
                "n_blocks": 1,
                "d_ff": 16,
                "patch_size": 2,
                "use_revin": True,
            },
            {
                "loss": QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
                "d_model": 16,
                "kernel_size": 3,
                "n_blocks": 1,
                "d_ff": 32,
                "patch_size": 4,
                "use_revin": False,
            },
            {
                "d_model": 16,
                "kernel_size": 7,
                "small_kernel_size": 3,
                "n_blocks": 1,
                "d_ff": 32,
                "patch_size": 4,
                "individual": True,
                "use_revin": False,
            },
        ]

        for param in params:
            dm_cfg = {"max_encoder_length": 8, "max_prediction_length": 2}
            dm_cfg.update(param.get("datamodule_cfg", {}))
            param["datamodule_cfg"] = dm_cfg

        return params
