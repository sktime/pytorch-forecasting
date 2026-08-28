"""Shared N-Beats adapter for pytorch-forecasting v2."""

from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.layers._nbeats._blocks import (
    NBEATSSeasonalBlock,
    NBEATSTrendBlock,
    SeasonalMixin,
    TrendMixin,
)
from pytorch_forecasting.metrics import Metric
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class NBeatsAdapterV2(TslibBaseModel):
    """Shared forward / training helpers for NBeats and NBeatsKAN (v2)."""

    def __init__(
        self,
        loss: Metric,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        backcast_loss_ratio: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            metadata=metadata,
        )
        self.backcast_loss_ratio = backcast_loss_ratio

    def _target_from_batch(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract univariate target history.

        v1 used ``x["encoder_cont"][..., 0]``. v2 tslib batches keep the target
        in ``history_target``.
        """
        target = x["history_target"]
        if target.ndim == 3:
            target = target[..., 0]
        return target

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pass forward of network.

        Network steps match v1 ``NBeatsAdapter.forward``; only input assembly
        and output packaging differ for the v2 API.
        """
        # --- v2 batch adapter (v1: target = x["encoder_cont"][..., 0]) ---
        target = self._target_from_batch(x)

        # --- same as v1 from here ---
        timesteps = self.context_length + self.prediction_length
        generic_forecast = [
            torch.zeros(
                (target.size(0), timesteps), dtype=torch.float32, device=self.device
            )
        ]
        trend_forecast = [
            torch.zeros(
                (target.size(0), timesteps), dtype=torch.float32, device=self.device
            )
        ]
        seasonal_forecast = [
            torch.zeros(
                (target.size(0), timesteps), dtype=torch.float32, device=self.device
            )
        ]
        forecast = torch.zeros(
            (target.size(0), self.prediction_length),
            dtype=torch.float32,
            device=self.device,
        )

        backcast = target  # initialize backcast
        for i, block in enumerate(self.net_blocks):
            # evaluate block
            backcast_block, forecast_block = block(backcast)

            # add for interpretation
            full = torch.cat([backcast_block.detach(), forecast_block.detach()], dim=1)
            if isinstance(block, (NBEATSTrendBlock, TrendMixin)):
                trend_forecast.append(full)
            elif isinstance(block, (NBEATSSeasonalBlock, SeasonalMixin)):
                seasonal_forecast.append(full)
            else:
                generic_forecast.append(full)

            # update backcast and forecast
            backcast = (
                backcast - backcast_block
            )  # do not use backcast -= backcast_block as this signifies an inline operation  # noqa: E501
            forecast = forecast + forecast_block

        prediction = forecast.unsqueeze(-1)
        backcast_out = (target - backcast).unsqueeze(-1)
        trend = torch.stack(trend_forecast, dim=0).sum(0).unsqueeze(-1)
        seasonality = torch.stack(seasonal_forecast, dim=0).sum(0).unsqueeze(-1)
        generic = torch.stack(generic_forecast, dim=0).sum(0).unsqueeze(-1)

        # v1 applied transform_output via BaseModel; v2 tslib does so when scales exist
        if "target_scale" in x:
            prediction = self.transform_output(prediction, x["target_scale"])
            backcast_out = self.transform_output(backcast_out, x["target_scale"])
            trend = self.transform_output(trend, x["target_scale"])
            seasonality = self.transform_output(seasonality, x["target_scale"])
            generic = self.transform_output(generic, x["target_scale"])

        # v1: to_network_output(...); v2: plain dict
        return {
            "prediction": prediction,
            "backcast": backcast_out,
            "trend": trend,
            "seasonality": seasonality,
            "generic": generic,
        }

    def training_step(
        self, batch: tuple[dict[str, torch.Tensor]], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Training step with optional backcast loss (v1 ``step`` parity)."""
        x, y = batch
        out = self(x)
        y_hat = out["prediction"]
        loss = self.loss(y_hat, y)

        if self.backcast_loss_ratio > 0:
            backcast = out["backcast"].squeeze(-1)
            encoder_target = self._target_from_batch(x)

            backcast_weight = (
                self.backcast_loss_ratio
                * self.prediction_length
                / max(self.context_length, 1)
            )
            backcast_weight = backcast_weight / (backcast_weight + 1)
            forecast_weight = 1 - backcast_weight

            # Compute backcast term directly (avoid Metric.update state / shape quirks).
            # v1 used self.loss(backcast, encoder_target); v2 BaseModel losses are
            # wired for forecast horizon shapes only.
            backcast_loss = (backcast - encoder_target).abs().mean() * backcast_weight
            loss = loss * forecast_weight + backcast_loss

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat, y, prefix="train")
        return {"loss": loss}
