"""NHiTS forecaster."""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class NHiTSForecaster(BaseForecaster):
    """N-HiTS Forecaster.

    Parameters
    ----------
    naive_level : bool, default=True
        Whether to start the residual loop from a naive (last-observation) level.
    shared_weights : bool, default=True
        Whether to share weights across blocks within each stack.
    activation : str, default="ReLU"
        Activation function for MLP layers.
    initialization : str, default="lecun_normal"
        Weight initialisation scheme.
    n_blocks : list of int, optional
        Number of blocks per stack. Defaults to ``[1, 1, 1]``.
    n_layers : int or list of int, default=2
        Number of fully-connected layers per block.
    hidden_size : int, default=512
        Width of the fully-connected layers inside each block.
    pooling_sizes : list of int, optional
        Pooling kernel sizes per stack. Derived from the prediction length if
        not given.
    downsample_frequencies : list of int, optional
        Downsampling factors for the forecast interpolation per stack. Derived
        from ``pooling_sizes`` if not given.
    pooling_mode : str, default="max"
        Pooling mode, one of ``"max"`` or ``"average"``.
    interpolation_mode : str, default="linear"
        Interpolation mode for the identity basis function.
    batch_normalization : bool, default=False
        Whether to apply batch normalisation in the MLP layers.
    dropout : float, default=0.0
        Dropout probability in the MLP layers.
    backcast_loss_ratio : float, default=0.0
        Weight of the backcast loss relative to the forecast loss.
    loss : Metric, optional
        Loss function for training. Defaults to ``MASE()``.
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
    [1] N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting
        (https://arxiv.org/abs/2201.12886).
    """

    _tags = {
        "info:name": "NHiTS_v2",
        "info:compute": 2,
        "authors": ["jdb78", "echo-xiao"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        naive_level: bool = True,
        shared_weights: bool = True,
        activation: str = "ReLU",
        initialization: str = "lecun_normal",
        n_blocks: list[int] | None = None,
        n_layers: int | list[int] = 2,
        hidden_size: int = 512,
        pooling_sizes: list[int] | None = None,
        downsample_frequencies: list[int] | None = None,
        pooling_mode: str = "max",
        interpolation_mode: str = "linear",
        batch_normalization: bool = False,
        dropout: float = 0.0,
        backcast_loss_ratio: float = 0.0,
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
        self.naive_level = naive_level
        self.shared_weights = shared_weights
        self.activation = activation
        self.initialization = initialization
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.pooling_sizes = pooling_sizes
        self.downsample_frequencies = downsample_frequencies
        self.pooling_mode = pooling_mode
        self.interpolation_mode = interpolation_mode
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.backcast_loss_ratio = backcast_loss_ratio
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
        NHiTS_v2 : type
            The model class.
        """
        from pytorch_forecasting.models.nhits._nhits_v2 import NHiTS_v2

        return NHiTS_v2

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
        """Kwargs for ``get_cls()``; ``loss=None`` resolves to MASE in the model."""
        return dict(
            naive_level=self.naive_level,
            shared_weights=self.shared_weights,
            activation=self.activation,
            initialization=self.initialization,
            n_blocks=self.n_blocks,
            n_layers=self.n_layers,
            hidden_size=self.hidden_size,
            pooling_sizes=self.pooling_sizes,
            downsample_frequencies=self.downsample_frequencies,
            pooling_mode=self.pooling_mode,
            interpolation_mode=self.interpolation_mode,
            batch_normalization=self.batch_normalization,
            dropout=self.dropout,
            backcast_loss_ratio=self.backcast_loss_ratio,
            loss=self.loss,
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
            Each dict is passed as keyword arguments to the forecaster.
            The key ``"datamodule_cfg"`` inside each dict is forwarded to
            the datamodule constructor.
        """
        from pytorch_forecasting.metrics import MAE, MASE, RMSE, SMAPE, QuantileLoss

        params = [
            {},
            dict(
                n_blocks=[1, 1, 1],
                n_layers=2,
                hidden_size=256,
                pooling_mode="max",
                interpolation_mode="linear",
                loss=RMSE(),
            ),
            dict(
                n_blocks=[1, 1],
                hidden_size=128,
                backcast_loss_ratio=0.1,
                loss=SMAPE(),
                logging_metrics=[MAE()],
            ),
            dict(
                n_blocks=[1],
                hidden_size=64,
                loss=MAE(),
            ),
            dict(
                n_blocks=[1],
                hidden_size=64,
                loss=QuantileLoss(),
                backcast_loss_ratio=0.0,
                logging_metrics=[MAE()],
            ),
            dict(
                n_blocks=[1],
                hidden_size=64,
                loss=MASE(),
            ),
        ]

        default_dm_cfg = {
            "max_encoder_length": 6,
            "max_prediction_length": 3,
        }

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            dm_cfg = default_dm_cfg.copy()
            dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = dm_cfg

        return params
