"""PatchTST forecaster."""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class PatchTSTForecaster(BaseForecaster):
    """PatchTST forecaster (v2).

    Parameters
    ----------
    enc_in: int, optional
        Number of input features. If not provided, it is inferred from data.
    hidden_size: int, default=16
        Dimension of the model embeddings.
    n_heads: int, default=2
        Number of attention heads.
    patch_len: int, default=16
        Length of each non-overlapping patch.
    stride: int, default=8
        Stride size for patching.
    padding: int, default=0
        Padding size for the input sequence.
    dropout: float, default=0.1
        Dropout rate.
    head_dropout: float, default=0.1
        Dropout rate for the output head.
    loss: nn.Module, optional
        Loss function to use for training. Defaults to ``MAE()``.
    logging_metrics: list[nn.Module] | None, default=None
        List of metrics to log.
    optimizer: Optimizer | str | None, default='adam'
        Optimizer to use for training.
    optimizer_params: dict | None, default=None
        Parameters for the optimizer.
    lr_scheduler: str | None, default=None
        Learning rate scheduler.
    lr_scheduler_params: dict | None, default=None
        Parameters for the scheduler.
    trainer : lightning.pytorch.Trainer, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    datamodule : EncoderDecoderTimeSeriesDataModule, optional
        Configured, data-less datamodule. See
        :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    ckpt_path : str or Path, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    """

    _tags = {
        "info:name": "PatchTST_v2",
        "info:compute": 3,
        "info:y_type": ["numeric"],
        "authors": ["nareshmethuku"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
        "tests:skip_by_name": ["test_integration"],
    }

    def __init__(
        self,
        enc_in: int | None = None,
        hidden_size: int = 16,
        n_heads: int = 2,
        patch_len: int = 16,
        stride: int = 8,
        padding: int = 0,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
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
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        self.head_dropout = head_dropout
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
        from pytorch_forecasting.models.patch_tst._patch_tst_v2 import PatchTST_v2

        return PatchTST_v2

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
            enc_in=self.enc_in,
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            patch_len=self.patch_len,
            stride=self.stride,
            padding=self.padding,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
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
        from pytorch_forecasting.data.encoders import GroupNormalizer

        return [
            {
                "hidden_size": 16,
                "n_heads": 2,
                "patch_len": 4,
                "stride": 4,
                "dropout": 0.1,
                "datamodule_cfg": {
                    "max_encoder_length": 16,
                    "max_prediction_length": 3,
                },
            },
            {
                "hidden_size": 32,
                "n_heads": 4,
                "patch_len": 8,
                "stride": 8,
                "dropout": 0.2,
                "datamodule_cfg": {
                    "max_encoder_length": 16,
                    "max_prediction_length": 3,
                },
            },
            {
                "hidden_size": 16,
                "n_heads": 2,
                "patch_len": 2,
                "stride": 2,
                "dropout": 0.1,
                "datamodule_cfg": {"max_encoder_length": 4, "max_prediction_length": 2},
            },
            {
                "hidden_size": 24,
                "n_heads": 3,
                "patch_len": 4,
                "stride": 2,
                "dropout": 0.15,
                "datamodule_cfg": dict(
                    max_encoder_length=6,
                    max_prediction_length=3,
                ),
            },
        ]
