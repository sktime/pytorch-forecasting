"""FreTS forecaster."""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class FreTSForecaster(BaseForecaster):
    """FreTS forecaster (v2).

    Parameters
    ----------
    embed_size : int, default=128
        Dimension of the learnable token embedding.
    hidden_size : int, default=256
        Hidden size of the FC output head.
    channel_independence : bool, default=True
        If True, each channel is processed independently (only temporal
        frequency mixing). If False, cross-channel frequency mixing is
        applied first.
    sparsity_threshold : float, default=0.01
        Soft-shrinkage threshold for frequency coefficient sparsity.
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
        "info:name": "FreTS",
        "info:compute": 2,
        "info:y_type": ["numeric"],
        "authors": ["echo-xiao"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        embed_size: int = 128,
        hidden_size: int = 256,
        channel_independence: bool = True,
        sparsity_threshold: float = 0.01,
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
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.channel_independence = channel_independence
        self.sparsity_threshold = sparsity_threshold
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
        FreTS : type
            The model class.
        """
        from pytorch_forecasting.models.frets._frets_v2 import FreTS

        return FreTS

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
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            channel_independence=self.channel_independence,
            sparsity_threshold=self.sparsity_threshold,
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
        from pytorch_forecasting.metrics import MAE, RMSE, SMAPE

        params = [
            {},
            dict(
                embed_size=32,
                hidden_size=64,
                channel_independence=True,
            ),
            dict(
                embed_size=64,
                hidden_size=128,
                channel_independence=False,
                logging_metrics=[SMAPE()],
            ),
            dict(
                embed_size=16,
                hidden_size=32,
                loss=MAE(),
            ),
            dict(
                embed_size=16,
                hidden_size=32,
                sparsity_threshold=0.0,
                loss=RMSE(),
            ),
            dict(
                embed_size=16,
                hidden_size=32,
                optimizer="adamw",
                lr_scheduler="cosine_annealing",
                lr_scheduler_params=dict(T_max=2),
            ),
            dict(
                embed_size=16,
                hidden_size=32,
                channel_independence=False,
                datamodule_cfg=dict(max_encoder_length=12, max_prediction_length=4),
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
