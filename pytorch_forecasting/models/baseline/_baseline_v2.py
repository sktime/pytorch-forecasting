"""Baseline forecaster model for PyTorch Forecasting v2."""

from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import MAE, Metric
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class Baseline_v2(BaseModel):
    """Baseline forecasting model that predicts using the last known target value.

    Parameters
    ----------
    loss : Metric, default=MAE()
        Loss function used during training.
    logging_metrics : list of nn.Module, optional
        Metrics logged during evaluation.
    optimizer : Optimizer or str, default="adam"
        Optimizer name or instance.
    optimizer_params : dict, optional
        Optimizer parameters.
    lr_scheduler : str, optional
        Learning rate scheduler name.
    lr_scheduler_params : dict, optional
        Learning rate scheduler parameters.
    metadata : dict, optional
        Metadata dictionary from the DataModule.
    """

    @classmethod
    def _pkg(cls):
        """Package container for the model."""
        from pytorch_forecasting.models.baseline._baseline_pkg_v2 import (
            Baseline_pkg_v2,
        )

        return Baseline_pkg_v2

    def __init__(
        self,
        loss: Metric = MAE(),
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self.metadata = metadata or {}
        self.max_prediction_length = self.metadata.get("max_prediction_length", 1)

    def forward(
        self,
        x: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass repeating last target value across forecast horizon.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input dictionary containing ``target_past``.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing predicted output tensor under key ``prediction``.
        """
        target_past = x.get("target_past")
        if target_past is None:
            raise KeyError("Input dictionary must contain key 'target_past'.")

        if target_past.ndim == 1:
            target_past = target_past.unsqueeze(-1)

        # Obtain last target value from past sequence
        last_target = target_past[:, -1, :]

        # Expand across prediction length
        prediction = last_target.unsqueeze(1).expand(-1, self.max_prediction_length, -1)

        return {"prediction": prediction}
