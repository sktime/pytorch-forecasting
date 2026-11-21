"""TIDE forecaster."""

from pathlib import Path
from typing import Any

from lightning.pytorch import Trainer
from torch import nn

from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


class TIDEForecaster(BaseForecaster):
    """TIDE forecaster (v2).

    Parameters
    ----------
    hidden_size : int, default=64
        Dimensionality of hidden layers in projections (R).
    d_model : int, default=32
        Dimensionality of model projections after feature projection (R̃).
    n_add_enc : int, default=1
        Number of additional encoder residual blocks (after the first).
    n_add_dec : int, default=1
        Number of additional decoder residual blocks (after the first).
    dropout_rate : float, default=0.1
        Dropout probability applied in residual blocks.
    activation : str, optional
        Name of activation function to use (e.g., ``"relu"``).
    embs : list of int, optional
        List specifying embedding sizes for categorical variables.
    persistence_weight : float, optional
        Weight for the persistence (autoregressive) component.
    optim : str or None, optional
        Name of optimizer (e.g., ``"adam"``), or None to use default.
    optim_config : dict or None, optional
        Optimizer configuration dictionary.
    scheduler_config : dict or None, optional
        Scheduler configuration dictionary.
    loss : nn.Module, optional
        Loss function module (e.g., ``MSELoss``, ``QuantileLoss``). Defaults to
        ``MAE()``.
    trainer : lightning.pytorch.Trainer, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    datamodule : EncoderDecoderTimeSeriesDataModule, optional
        Configured, data-less datamodule. See
        :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    ckpt_path : str or Path, optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.
    """

    _tags = {
        "info:name": "TIDE",
        "authors": ["fbk_dsipts", "phoeenniixx"],
        "info:compute": 3,
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        hidden_size: int = 64,
        d_model: int = 32,
        n_add_enc: int = 1,
        n_add_dec: int = 1,
        dropout_rate: float = 0.1,
        activation: str = "",
        embs: list[int] | None = None,
        persistence_weight: float = 0.0,
        optim: str | None = None,
        optim_config: dict | None = None,
        scheduler_config: dict | None = None,
        loss: nn.Module | None = None,
        trainer: Trainer | None = None,
        datamodule: Any = None,
        ckpt_path: str | Path | None = None,
    ):
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.n_add_enc = n_add_enc
        self.n_add_dec = n_add_dec
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.embs = embs
        self.persistence_weight = persistence_weight
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.loss = loss
        self.trainer = trainer
        self.datamodule = datamodule
        self.ckpt_path = ckpt_path
        super().__init__(
            trainer=self.trainer, datamodule=self.datamodule, ckpt_path=self.ckpt_path
        )

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.tide._tide_dsipts._tide_v2 import TIDE

        return TIDE

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
            d_model=self.d_model,
            n_add_enc=self.n_add_enc,
            n_add_dec=self.n_add_dec,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            embs=[] if self.embs is None else self.embs,
            persistence_weight=self.persistence_weight,
            optim=self.optim,
            optim_config=self.optim_config,
            scheduler_config=self.scheduler_config,
            loss=MAE() if self.loss is None else self.loss,
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
        import torch.nn as nn

        from pytorch_forecasting.metrics import MAE, MAPE

        params = [
            dict(
                hidden_size=16,
                d_model=8,
                n_add_enc=1,
                n_add_dec=1,
                dropout_rate=0.1,
                loss=nn.MSELoss(),
            ),
            dict(
                hidden_size=32,
                d_model=16,
                n_add_enc=2,
                n_add_dec=2,
                dropout_rate=0.2,
                datamodule_cfg=dict(max_encoder_length=5, max_prediction_length=3),
                loss=MAE(),
            ),
            dict(
                hidden_size=64,
                d_model=32,
                n_add_enc=3,
                n_add_dec=2,
                dropout_rate=0.1,
                datamodule_cfg=dict(max_encoder_length=4, max_prediction_length=2),
                loss=MAPE(),
            ),
        ]
        default_dm_cfg = {"max_encoder_length": 4, "max_prediction_length": 3}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)

            param["datamodule_cfg"] = default_dm_cfg

        return params
