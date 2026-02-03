"""
N-BEATS v2 model for time series forecasting without covariates.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers._nbeats._blocks import (
    NBEATSGenericBlock,
    NBEATSSeasonalBlock,
    NBEATSTrendBlock,
)
from pytorch_forecasting.metrics import MASE, Metric
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class NBEATS(BaseModel):
    @classmethod
    def _pkg(cls):
        from pytorch_forecasting.models.nbeats._nbeats_v2_pkg import (
            NBEATS_pkg_v2,
        )

        return NBEATS_pkg_v2

    def __init__(
        self,
        *,
        stack_types: list[str] | None = None,
        num_blocks: list[int] | None = None,
        num_block_layers: list[int] | None = None,
        widths: list[int] | None = None,
        sharing: list[bool] | None = None,
        expansion_coefficient_lengths: list[int] | None = None,
        dropout: float = 0.0,
        loss: Metric | None = None,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs,
    ):
        if loss is None:
            loss = MASE()

        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self.metadata = metadata

        self.context_length = metadata["max_encoder_length"]
        self.prediction_length = metadata["max_prediction_length"]

        self.stack_types = stack_types or ["trend", "seasonality"]
        self.num_blocks = num_blocks or [3, 3]
        self.num_block_layers = num_block_layers or [3, 3]
        self.widths = widths or [32, 512]
        self.sharing = sharing or [True, True]
        self.expansion_coefficient_lengths = expansion_coefficient_lengths or [3, 7]
        self.dropout = dropout

        self._init_network()

    def _make_block(self, stack_id: int, stack_type: str) -> nn.Module:
        if stack_type == "generic":
            return NBEATSGenericBlock(
                units=self.widths[stack_id],
                thetas_dim=self.expansion_coefficient_lengths[stack_id],
                num_block_layers=self.num_block_layers[stack_id],
                backcast_length=self.context_length,
                forecast_length=self.prediction_length,
                dropout=self.dropout,
            )
        if stack_type == "seasonality":
            return NBEATSSeasonalBlock(
                units=self.widths[stack_id],
                num_block_layers=self.num_block_layers[stack_id],
                backcast_length=self.context_length,
                forecast_length=self.prediction_length,
                min_period=self.expansion_coefficient_lengths[stack_id],
                dropout=self.dropout,
            )
        if stack_type == "trend":
            return NBEATSTrendBlock(
                units=self.widths[stack_id],
                thetas_dim=self.expansion_coefficient_lengths[stack_id],
                num_block_layers=self.num_block_layers[stack_id],
                backcast_length=self.context_length,
                forecast_length=self.prediction_length,
                dropout=self.dropout,
            )
        raise ValueError(f"Unknown stack type: {stack_type}")

    def _init_network(self) -> None:
        self.net_blocks = nn.ModuleList()

        for stack_id, stack_type in enumerate(self.stack_types):
            if self.sharing[stack_id]:
                block = self._make_block(stack_id, stack_type)
                for _ in range(self.num_blocks[stack_id]):
                    self.net_blocks.append(block)
            else:
                for _ in range(self.num_blocks[stack_id]):
                    self.net_blocks.append(self._make_block(stack_id, stack_type))

    def _expand_for_quantiles(self, prediction: torch.Tensor) -> torch.Tensor:
        quantiles = getattr(self.loss, "quantiles", None)
        if not quantiles:
            return prediction

        return prediction.repeat(1, 1, len(quantiles))

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        target = x["target_past"].squeeze(-1)
        batch_size = target.size(0)

        backcast = target
        forecast = torch.zeros(
            batch_size,
            self.prediction_length,
            device=target.device,
        )

        trend_parts, seasonality_parts, generic_parts = [], [], []

        for block in self.net_blocks:
            backcast_block, forecast_block = block(backcast)

            if isinstance(block, NBEATSTrendBlock):
                trend_parts.append(torch.cat([backcast_block, forecast_block], dim=1))
            elif isinstance(block, NBEATSSeasonalBlock):
                seasonality_parts.append(
                    torch.cat([backcast_block, forecast_block], dim=1)
                )
            else:
                generic_parts.append(torch.cat([backcast_block, forecast_block], dim=1))

            backcast = backcast - backcast_block
            forecast = forecast + forecast_block

        prediction = forecast.unsqueeze(-1)
        prediction = self._expand_for_quantiles(prediction)

        explained_backcast = (target - backcast).unsqueeze(-1)

        def _empty():
            return torch.zeros(
                batch_size,
                self.context_length + self.prediction_length,
                1,
                device=target.device,
            )

        return {
            "prediction": prediction,
            "backcast": explained_backcast,
            "trend": (
                torch.stack(trend_parts).sum(0).unsqueeze(-1)
                if trend_parts
                else _empty()
            ),
            "seasonality": (
                torch.stack(seasonality_parts).sum(0).unsqueeze(-1)
                if seasonality_parts
                else _empty()
            ),
            "generic": (
                torch.stack(generic_parts).sum(0).unsqueeze(-1)
                if generic_parts
                else _empty()
            ),
        }
