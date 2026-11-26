"""Naive forecaster."""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Naive Forecaster.

    Baseline forecaster that repeats the last observed target value across
    the prediction horizon.

    Parameters
    ----------
    loss : nn.Module, optional
        Loss function used for metric calculation and logging.
        Defaults to ``MAE()``.
    logging_metrics : list[nn.Module] | None, default=None
        Metrics to log during training, validation, and testing.
    trainer : lightning.pytorch.Trainer, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    datamodule : TslibDataModule, optional
        Configured, data-less datamodule. See
        :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    ckpt_path : str or Path, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    """

    _tags = {
        "info:name": "Naive",
        "info:compute": 1,
        "authors": ["Utkarshhs"],
        "info:y_type": ["numeric"],
        "capability:exogenous": False,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        loss: nn.Module | None = None,
        logging_metrics: list[nn.Module] | None = None,
        trainer: Trainer | None = None,
        datamodule: Any = None,
        ckpt_path: str | Path | None = None,
    ):
        self.loss = loss
        self.logging_metrics = logging_metrics
        self.trainer = trainer
        self.datamodule = datamodule
        self.ckpt_path = ckpt_path
        super().__init__(
            trainer=self.trainer, datamodule=self.datamodule, ckpt_path=self.ckpt_path
        )

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.baseline._naive_v2 import Naive

        return Naive

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import TslibDataModule

        return TslibDataModule

    def get_model_params(self) -> dict[str, Any]:
        """Kwargs for ``get_cls()``, with ``None`` sentinels resolved."""
        from pytorch_forecasting.metrics import MAE

        return dict(
            loss=MAE() if self.loss is None else self.loss,
            logging_metrics=self.logging_metrics,
        )

    @classmethod
    def get_test_train_params(cls):
        """
        Return testing parameter settings for the trainer.

        Returns
        -------
        list[dict]
            Parameter configurations used to create testing instances of the NAIVE
            class.
        """

        import torch.nn as nn

        from pytorch_forecasting.metrics import SMAPE

        params = [
            {},
            dict(
                logging_metrics=[SMAPE()],
            ),
            dict(
                loss=nn.MSELoss(),
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
