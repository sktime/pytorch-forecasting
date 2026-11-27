"""
FiLM forecaster.
"""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class FiLMForecaster(BaseForecaster):
    """FiLM forecaster (v2).

    Parameters
    ----------
    window_size: int, default=256
        Order of the HiPPO-LegT projection (number of Legendre
        polynomials).
    modes: int, default=32
        Maximum number of lowest Fourier modes kept in the spectral
        convolution layers.
    multiscale: tuple[int, ...] or list[int], default=(1, 2, 4)
        Multiples of ``prediction_length`` defining the input horizon of
        each spectral branch.
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
        "info:name": "FiLM",
        "info:compute": 2,
        "authors": ["thuml"],
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        window_size: int = 256,
        modes: int = 32,
        multiscale: tuple[int, ...] | list[int] = (1, 2, 4),
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
        self.window_size = window_size
        self.modes = modes
        self.multiscale = multiscale
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
        from pytorch_forecasting.models.film._film_v2 import FiLM

        return FiLM

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import TslibDataModule

        return TslibDataModule

    def get_model_params(self) -> dict[str, Any]:
        """Kwargs for ``get_cls()``, with ``None`` sentinels resolved."""
        from pytorch_forecasting.metrics import MAE

        return dict(
            window_size=self.window_size,
            modes=self.modes,
            multiscale=self.multiscale,
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
            Each dict is passed as keyword arguments to the forecaster.
            The key ``"datamodule_cfg"`` inside each dict is forwarded to
            the datamodule constructor.
        """
        import torch.nn as nn

        from pytorch_forecasting.metrics import QuantileLoss, SMAPE

        params = [
            {},
            dict(window_size=16, modes=4),
            dict(window_size=8, modes=2, multiscale=(1, 2)),
            dict(
                window_size=8,
                modes=2,
                logging_metrics=[SMAPE()],
            ),
            dict(
                window_size=8,
                modes=2,
                loss=nn.MSELoss(),
            ),
            dict(
                window_size=8,
                modes=2,
                multiscale=(1,),
                datamodule_cfg=dict(context_length=16, prediction_length=4),
            ),
            dict(
                window_size=8,
                modes=2,
                loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            ),
            dict(
                window_size=8,
                modes=2,
                datamodule_cfg=dict(
                    context_length=12,
                    prediction_length=3,
                    add_relative_time_idx=False,
                ),
            ),
            dict(
                optimizer="adamw",
                loss=nn.HuberLoss(),
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
            dm_cfg = default_dm_cfg.copy()
            dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = dm_cfg

        return params
