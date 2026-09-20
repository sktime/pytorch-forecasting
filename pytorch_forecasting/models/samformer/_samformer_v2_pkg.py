"""
Samformer package container.
"""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class SamformerForecaster(BaseForecaster):
    """Samformer forecaster.

    Parameters
    ----------
    hidden_size : int, default=32
        Hidden size of the attention projections.
    use_revin : bool, default=True
        Whether to apply reversible instance normalisation.
    out_channels : int, default=1
        Number of output channels; has to be 1 (no MultiLoss support in v2).
    persistence_weight : float, default=0.0
        Weight of the persistence (last value) baseline.
    loss : nn.Module, optional
        Loss to optimise. Defaults to ``MAE()``.
    logging_metrics : list of nn.Module, optional
        Metrics to log during training.
    optimizer : Optimizer or str, default="adam"
        Optimizer, or its name.
    optimizer_params : dict, optional
        Keyword arguments for the optimizer.
    lr_scheduler : str, optional
        Learning-rate scheduler name.
    lr_scheduler_params : dict, optional
        Keyword arguments for the scheduler.
    trainer : lightning.pytorch.Trainer, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    datamodule : EncoderDecoderTimeSeriesDataModule, optional
        Configured, data-less datamodule. See
        :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    """

    _tags = {
        "info:name": "Samformer",
        "authors": ["fbk_dsipts", "PranavBhatP"],
        "info:compute": 2,
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "capability:multivariate": False,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        hidden_size: int = 32,
        use_revin: bool = True,
        out_channels: int = 1,
        persistence_weight: float = 0.0,
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
        self.hidden_size = hidden_size
        self.use_revin = use_revin
        self.out_channels = out_channels
        self.persistence_weight = persistence_weight
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
        from pytorch_forecasting.models.samformer._samformer_v2 import Samformer

        return Samformer

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
            hidden_size=self.hidden_size,
            use_revin=self.use_revin,
            out_channels=self.out_channels,
            persistence_weight=self.persistence_weight,
            loss=MAE() if self.loss is None else self.loss,
            logging_metrics=self.logging_metrics,
            optimizer=self.optimizer,
            optimizer_params=self.optimizer_params,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_params=self.lr_scheduler_params,
        )

    @classmethod
    def get_test_train_params(cls):
        """Return testing parameters settings for the trainer.

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
            {
                "loss": nn.MSELoss(),
                "hidden_size": 32,
                "use_revin": False,
            },
            {
                "hidden_size": 16,
                "use_revin": True,
                "out_channels": 1,
                "persistence_weight": 0.0,
            },
            {
                "loss": QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
                "hidden_size": 32,
                "use_revin": False,
            },
        ]

        default_dm_cfg = {"max_encoder_length": 4, "max_prediction_length": 3}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)

            param["datamodule_cfg"] = default_dm_cfg

        return params
