"""
Naive baseline model for PyTorch Forecasting v2.
------------------------------------------------
"""

#################################################
# NOTE: This is an experimental implementation  #
# of Naive for PyTorch Forecasting v2.          #
# It is an unstable API and subject to change.  #
#################################################

from typing import Any

import torch
import torch.nn as nn

from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class Naive(TslibBaseModel):
    """
    Naive baseline model that repeats the last observed target value.

    Parameters
    ----------
    loss : nn.Module
        Loss function used for metric calculation and logging.
    logging_metrics : list[nn.Module] | None, default=None
        List of metrics to log during training, validation, and testing.
    metadata : dict | None, default=None
        Metadata for the model from ``TslibDataModule``.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.baseline._naive_forecaster_v2 import (
            NaiveForecaster,
        )

        return NaiveForecaster

    def __init__(
        self,
        loss: nn.Module,
        logging_metrics: list[nn.Module] | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            metadata=metadata,
        )

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        pass  # no network to initialize for naive model

    def configure_optimizers(self):
        return None

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass repeating the last observed target value.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing input tensors, including ``"history_target"``.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the model prediction under ``"prediction"``.
        """

        if "history_target" not in x or x["history_target"].size(1) == 0:
            raise ValueError(
                "Input dictionary must contain non-empty 'history_target'."
            )

        history_target = x["history_target"]
        last_value = history_target[:, -1:, :]
        prediction = last_value.repeat(1, self.prediction_length, 1)

        if "target_scale" in x and hasattr(self, "transform_output"):
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}

    def training_step(self, batch, batch_idx):
        super().training_step(batch, batch_idx)
        return None
