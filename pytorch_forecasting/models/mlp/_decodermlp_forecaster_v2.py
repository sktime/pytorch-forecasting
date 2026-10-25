"""DecoderMLP forecaster."""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class DecoderMLPForecaster(BaseForecaster):
    """DecoderMLP forecaster (v2).

    Parameters
    ----------
    hidden_size : int, default=300
        Hidden layer width of the MLP.
    n_hidden_layers : int, default=3
        Number of hidden layers.
    dropout : float, default=0.1
        Dropout probability.
    norm : bool, default=True
        Whether to apply ``LayerNorm`` in the MLP.
    activation_class : str, default="ReLU"
        Name of a ``torch.nn`` activation class.
    loss : nn.Module, optional
        Loss function for training. Defaults to ``MAE()``.
    logging_metrics : Optional[list[nn.Module]], default=None
        Metrics to log during training, validation, and testing.
    optimizer : Optional[Union[Optimizer, str]], default="adam"
        Optimizer to use for training.
    optimizer_params : Optional[dict], default=None
        Parameters for the optimizer.
    lr_scheduler : Optional[str], default=None
        Learning rate scheduler to use.
    lr_scheduler_params : Optional[dict], default=None
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
        "info:name": "DecoderMLP_v2",
        "info:compute": 1,
        "authors": ["jdb78", "echo-xiao"],
        # TODO: the v2 datamodule supports categorical inputs but not
        # categorical targets yet; add "categorical" to y_type once
        # EncoderDecoderTimeSeriesDataModule supports categorical targets.
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "capability:multivariate": False,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": True,
    }

    def __init__(
        self,
        hidden_size: int = 300,
        n_hidden_layers: int = 3,
        dropout: float = 0.1,
        norm: bool = True,
        activation_class: str = "ReLU",
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
        self.n_hidden_layers = n_hidden_layers
        self.dropout = dropout
        self.norm = norm
        self.activation_class = activation_class
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
        from pytorch_forecasting.models.mlp._decodermlp_v2 import DecoderMLP_v2

        return DecoderMLP_v2

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
            n_hidden_layers=self.n_hidden_layers,
            dropout=self.dropout,
            norm=self.norm,
            activation_class=self.activation_class,
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
            Parameters to create testing instances of the class. Each dict is passed
            as ``model_cfg`` to the package constructor; the ``"datamodule_cfg"`` key
            is forwarded to the datamodule constructor.
        """
        from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, QuantileLoss

        params = [
            {},
            dict(
                hidden_size=64, n_hidden_layers=2, dropout=0.1, norm=True, loss=RMSE()
            ),
            dict(
                hidden_size=128,
                n_hidden_layers=1,
                activation_class="ReLU",
                loss=SMAPE(),
                logging_metrics=[MAE()],
            ),
            dict(hidden_size=32, n_hidden_layers=2, norm=False, loss=MAE()),
            dict(hidden_size=64, n_hidden_layers=1, loss=QuantileLoss()),
            dict(
                optimizer="adamw",
                lr_scheduler="cosine_annealing",
                lr_scheduler_params={"T_max": 5},
                loss=MAE(),
            ),
        ]

        default_dm_cfg = {"max_encoder_length": 4, "max_prediction_length": 3}
        for param in params:
            dm_cfg = default_dm_cfg.copy()
            dm_cfg.update(param.get("datamodule_cfg", {}))
            param["datamodule_cfg"] = dm_cfg

        return params
