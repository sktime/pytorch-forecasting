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

    def __init__(
        self,
        loss: nn.Module,
        # specific params
        hidden_size: int,
        use_revin: bool,
        out_channels: Optional[Union[int, list[int]]] = 1,
        persistence_weight: float = 0.0,
        logging_metrics: Optional[list[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[dict] = None,
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

        self.max_encoder_length = self.metadata["max_encoder_length"]
        self.max_prediction_length = self.metadata["max_prediction_length"]
        self.encoder_cont = self.metadata["encoder_cont"]
        self.encoder_input_dim = self.encoder_cont
        self.decoder_cont = self.metadata["decoder_cont"]
        self.decoder_input_dim = self.decoder_cont

        self.hidden_size = hidden_size
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
        input_tensor = x.get("encoder_cont")

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

        if self.use_revin:
            out = self.revin(out, mode="denorm")

        prediction = out.transpose(1, 2)  # (batch, timesteps, channels)

        prediction = prediction[..., self.output_channel_indices]

        # single target case.
        if isinstance(self.out_channels, int) and self.out_channels == 1:
            if prediction.shape[-1] != 1:
                prediction = prediction[..., 0]
            return {"predictions": prediction}

        # multi-target case.
        elif isinstance(self.out_channels, list):
            predictions_list = []
            target_pred = prediction[..., self.out_channels]
            for i in range(len(self.out_channels)):
                target_pred_i = target_pred[..., i]
                predictions_list.append(target_pred_i)

            return {"prediction": predictions_list}
