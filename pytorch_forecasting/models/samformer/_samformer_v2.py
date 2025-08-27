"""
Samformer Model from DSIPTS for PyTorch Forecasting
---------------------------------------------------
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers import RevIN
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class Samformer(BaseModel):
    """
    Samformer: Unlocking the Potential of Transformers in Time Series Forecasting
    with Sharpness-Aware Minimization and Channel-Wise Attention.

    Parameters
    ----------
    out_channels : int, optional
        Number of variables to be predicted. Default is 1.
    hidden_size : int, optional
        First embedding size of the model ('r' in the paper). Default is 512.
    use_revin : bool, optional
        Whether to use Reverse Instance Normalization. Default is True.
    persistence_weight : float, optional
        Weight for persistence baseline. Default is 0.0.
    """

    @classmethod
    def _pkg(cls):
        """Return the package class for this model."""
        from pytorch_forecasting.models.samformer._samformer_v2_pkg import (
            Samformer_pkg_v2,
        )

        return Samformer_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        # specific params
        hidden_size: int,
        use_revin: bool,
        # out_channels has to be 1, due to lack of MultiLoss support in v2.
        out_channels: Optional[Union[int, list[int]]] = 1,
        persistence_weight: float = 0.0,
        logging_metrics: Optional[list[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
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

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "optimizer"])
        self.metadata = metadata
        self.n_quantiles = 1

        if hasattr(loss, "quantiles") and loss.quantiles is not None:
            self.n_quantiles = len(loss.quantiles)

        self.max_encoder_length = self.metadata["max_encoder_length"]
        self.max_prediction_length = self.metadata["max_prediction_length"]
        self.encoder_cont = self.metadata["encoder_cont"]
        self.encoder_input_dim = self.encoder_cont + 1  # +1 for target variable input.

        self.hidden_size = hidden_size
        if out_channels != 1:
            raise ValueError(
                "out_channels has to be 1 for Samformer,",
                " due to lack of MultiLoss support in v2.",
            )
        self.out_channels = out_channels
        self.use_revin = use_revin
        self.persistence_weight = persistence_weight

        if self.use_revin:
            self.revin = RevIN(num_features=self.encoder_input_dim)

        self.compute_keys = nn.Linear(self.max_encoder_length, self.hidden_size)
        self.compute_queries = nn.Linear(self.max_encoder_length, self.hidden_size)
        self.compute_values = nn.Linear(
            self.max_encoder_length, self.max_encoder_length
        )  # noqa: E501
        self.linear_forecaster = nn.Linear(
            self.max_encoder_length, self.max_prediction_length
        )  # noqa: E501

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input data containing past and future sequences.

        Returns
        -------
        dict[str, torch.Tensor]
            Output predictions.
        """
        encoder_cont = x["encoder_cont"]
        target = x["target_past"]
        input_tensor = torch.cat((encoder_cont, target), dim=-1)
        # batch_size = input_tensor.shape[0]

        if self.use_revin:
            x_norm = self.revin(input_tensor, mode="norm").transpose(1, 2)
        else:
            x_norm = input_tensor.transpose(1, 2)

        queries = self.compute_queries(x_norm)
        keys = self.compute_keys(x_norm)
        values = self.compute_values(x_norm)

        att_score = nn.functional.scaled_dot_product_attention(queries, keys, values)

        out = x_norm + att_score
        out = self.linear_forecaster(out)

        out = out.transpose(1, 2)

        target_predictions = out[:, :, -1]  # (batch_size, max_prediction_length)

        if self.n_quantiles > 1:
            target_predictions = target_predictions.expand(-1, -1, self.n_quantiles)
        elif self.n_quantiles == 1:
            target_predictions = target_predictions.unsqueeze(-1)
        return {"prediction": target_predictions}
