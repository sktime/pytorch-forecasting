"""SCINet forecaster."""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class SCINetForecaster(BaseForecaster):
    """SCINet forecaster (v2).

    Parameters
    ----------
    num_stacks : int, default=1
        Number of stacked SCITree modules.
    num_levels : int, default=3
        Depth of the binary decomposition tree.
        Input sequence length must satisfy
        ``context_length % (2 ** num_levels) == 0``.
    hid_size : int, default=1
        Channel expansion factor for the hidden conv layers inside
        each SCI-Block.  Hidden channels = n_channels * hid_size.
    kernel_size : int, default=5
        Kernel width for all Conv1d layers.
    dropout : float, default=0.5
        Dropout probability inside each SCI-Block.
    loss : Metric, optional
        Loss to optimise. Defaults to
        :class:`~pytorch_forecasting.metrics.MAE`.
    logging_metrics : list of nn.Module, optional
        Additional metrics logged during training and validation.
    optimizer : Optimizer or str, optional
        Optimizer used for training. Default is ``"adam"``.
    optimizer_params : dict, optional
        Parameters forwarded to the optimizer constructor.
    lr_scheduler : str, optional
        Learning rate scheduler name.
    lr_scheduler_params : dict, optional
        Parameters forwarded to the LR scheduler constructor.
    trainer : lightning.pytorch.Trainer, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    datamodule : EncoderDecoderTimeSeriesDataModule, optional
        Configured, data-less datamodule. See
        :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    ckpt_path : str or Path, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    """

    _tags = {
        "info:name": "SCINet_v2",
        "info:compute": 2,
        "authors": ["echo-xiao"],
        "capability:exogenous": False,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        num_stacks: int = 1,
        num_levels: int = 3,
        hid_size: int = 1,
        kernel_size: int = 5,
        dropout: float = 0.5,
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
        self.num_stacks = num_stacks
        self.num_levels = num_levels
        self.hid_size = hid_size
        self.kernel_size = kernel_size
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
        super().__init__(
            trainer=self.trainer, datamodule=self.datamodule, ckpt_path=self.ckpt_path
        )

    @classmethod
    def get_cls(cls):
        """Get model class.

        Returns
        -------
        SCINet : type
            The model class.
        """
        from pytorch_forecasting.models.scinet._scinet_v2 import SCINet_v2

        return SCINet_v2

    @classmethod
    def get_datamodule_cls(cls):
        """Get datamodule class used for training.

        Returns
        -------
        EncoderDecoderTimeSeriesDataModule : type
            The datamodule class.
        """
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    def get_model_params(self) -> dict[str, Any]:
        """Kwargs for ``get_cls()``, with ``None`` sentinels resolved."""
        from pytorch_forecasting.metrics import MAE

        return dict(
            num_stacks=self.num_stacks,
            num_levels=self.num_levels,
            hid_size=self.hid_size,
            kernel_size=self.kernel_size,
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
        """Return testing parameter settings for the trainer.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
            Each dict is passed as ``model_cfg`` to the package constructor.
            The key ``"datamodule_cfg"`` inside each dict is forwarded to
            the datamodule constructor.
        """
        from pytorch_forecasting.metrics import MAE, SMAPE

        params = [
            dict(
                datamodule_cfg={
                    "max_encoder_length": 8,
                    "max_prediction_length": 4,
                }
            ),
            dict(
                num_stacks=2,
                num_levels=2,
                hid_size=2,
                datamodule_cfg={
                    "max_encoder_length": 8,
                    "max_prediction_length": 4,
                },
            ),
            dict(
                num_levels=1,
                kernel_size=3,
                dropout=0.1,
                logging_metrics=[SMAPE()],
                datamodule_cfg={
                    "max_encoder_length": 8,
                    "max_prediction_length": 4,
                },
            ),
            dict(
                num_levels=2,
                loss=MAE(),
                datamodule_cfg={
                    "max_encoder_length": 8,
                    "max_prediction_length": 4,
                },
            ),
        ]

        return params
