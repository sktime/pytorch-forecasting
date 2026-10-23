"""xLSTMTime forecaster."""

from pathlib import Path
from typing import Any, Literal

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class xLSTMTimeForecaster(BaseForecaster):
    """xLSTMTime forecaster (v2).

    Parameters
    ----------
    hidden_size : int, default 32
        Hidden size of the xLSTM network; also used by batch norm / LSTM internals.
    xlstm_type : {"slstm", "mlstm"}, default "slstm"
        Specifies which xLSTM variant to use:
        - "slstm": stabilized LSTM with scalar memory,
        - "mlstm": matrix-memory variant for higher capacity and scalability.
    num_layers : int, default 1
        Number of recurrent layers in the sLSTM or mLSTM network.
    decomposition_kernel : int, default 25
        Kernel size for series decomposition into trend and seasonal components.
    input_projection_size : int, optional
        If specified, the encoded input (trend + seasonal) is projected to this size
        before being fed to the xLSTM; otherwise equals hidden_size.
    dropout : float, default 0.1
        Dropout rate applied within the recurrent layers.
    loss : nn.Module, optional
        Loss (and evaluation metric) used during training. Defaults to ``MAE()``.
    logging_metrics : list of nn.Module, optional
        Metrics logged during training / validation / testing.
    optimizer : Optimizer or str, optional
        Optimizer used for training.
    optimizer_params : dict, optional
        Parameters for the optimizer.
    lr_scheduler : str, optional
        Learning rate scheduler name.
    lr_scheduler_params : dict, optional
        Parameters for the learning rate scheduler.
    trainer : lightning.pytorch.Trainer, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    datamodule : EncoderDecoderTimeSeriesDataModule, optional
        Configured, data-less datamodule. See
        :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    ckpt_path : str or Path, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    """

    _tags = {
        "info:name": "xLSTMTime",
        "info:compute": 3,
        "info:y_type": ["numeric"],
        "authors": ["muslehal", "phoeenniixx", "Faakhir30"],
        "capability:exogenous": True,
        "capability:multivariate": False,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        hidden_size: int = 32,
        xlstm_type: Literal["slstm", "mlstm"] = "slstm",
        num_layers: int = 1,
        decomposition_kernel: int = 25,
        input_projection_size: int | None = None,
        dropout: float = 0.1,
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
        self.xlstm_type = xlstm_type
        self.num_layers = num_layers
        self.decomposition_kernel = decomposition_kernel
        self.input_projection_size = input_projection_size
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
        """Get model class."""
        from pytorch_forecasting.models.xlstm._xlstm_v2 import xLSTMTime_v2

        return xLSTMTime_v2

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
            xlstm_type=self.xlstm_type,
            num_layers=self.num_layers,
            decomposition_kernel=self.decomposition_kernel,
            input_projection_size=self.input_projection_size,
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
        """Return testing parameter settings for the trainer."""
        from pytorch_forecasting.metrics import MAE, MAPE, QuantileLoss

        params = [
            {},
            {"xlstm_type": "mlstm"},
            {"num_layers": 2},
            {"xlstm_type": "slstm", "input_projection_size": 32},
            {
                "xlstm_type": "mlstm",
                "decomposition_kernel": 3,
                "dropout": 0.2,
                "loss": MAE(),
            },
            {
                "loss": QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
                "hidden_size": 16,
            },
            {
                "optimizer": "adamw",
                "lr_scheduler": "cosine_annealing",
                "lr_scheduler_params": {"T_max": 5},
            },
        ]

        default_dm_cfg = {
            "max_encoder_length": 8,
            "max_prediction_length": 3,
        }

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = default_dm_cfg.copy()

        return params
