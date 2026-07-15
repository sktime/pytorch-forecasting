"""
N-Beats model for timeseries forecasting without covariates.
"""

from typing import Any, Optional, Union
import warnings

from lightning.pytorch.utilities.types import STEP_OUTPUT
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


class NBeats(BaseModel):
    """
    Initialize NBeats Model v2.

    Based on the article
    `N-BEATS: Neural basis expansion analysis for interpretable time series
        forecasting <http://arxiv.org/abs/1905.10437>`_.
    """

    @classmethod
    def _pkg(cls):
        """Package for the model."""
        from pytorch_forecasting.models.nbeats._nbeats_pkg_v2 import NBeats_pkg_v2

        return NBeats_pkg_v2

    def __init__(
        self,
        loss: Metric,
        stack_types: list[str] | None = None,
        num_blocks: list[int] | None = None,
        num_block_layers: list[int] | None = None,
        widths: list[int] | None = None,
        sharing: list[bool] | None = None,
        expansion_coefficient_lengths: list[int] | None = None,
        dropout: float = 0.1,
        backcast_loss_ratio: float = 0.0,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ):
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        elif dropout > 0.3:
            warnings.warn("dropout is greater than 0.3, clipping to 0.3", UserWarning)
            dropout = 0.3

        if backcast_loss_ratio < 0.0:
            raise ValueError("backcast_loss_ratio must be non-negative")

        if expansion_coefficient_lengths is None:
            expansion_coefficient_lengths = [3, 7]
        if sharing is None:
            sharing = [True, True]
        if widths is None:
            widths = [32, 512]
        if num_block_layers is None:
            num_block_layers = [3, 3]
        if num_blocks is None:
            num_blocks = [3, 3]
        if stack_types is None:
            stack_types = ["trend", "seasonality"]

        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )

        warnings.warn(
            "NBeats is an experimental model implemented on BaseModelV2. "
            "It is an unstable version and may be subject to unannounced changes. "
            "Please use with caution.",
            UserWarning,
        )

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self.metadata = metadata or {}

        self.context_length = self.metadata.get(
            "max_encoder_length", self.metadata.get("context_length", 1)
        )
        self.prediction_length = self.metadata.get(
            "max_prediction_length", self.metadata.get("prediction_length", 1)
        )

        self.n_quantiles = 1
        if hasattr(loss, "quantiles") and loss.quantiles is not None:
            self.n_quantiles = len(loss.quantiles)

        # setup stacks
        self.net_blocks = nn.ModuleList()
        for stack_id, stack_type in enumerate(stack_types):
            for _ in range(num_blocks[stack_id]):
                if stack_type == "generic":
                    net_block = NBEATSGenericBlock(
                        units=self.hparams.widths[stack_id],
                        thetas_dim=self.hparams.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        dropout=dropout,
                    )
                elif stack_type == "seasonality":
                    net_block = NBEATSSeasonalBlock(
                        units=self.hparams.widths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        min_period=expansion_coefficient_lengths[stack_id],
                        dropout=dropout,
                    )
                elif stack_type == "trend":
                    net_block = NBEATSTrendBlock(
                        units=self.hparams.widths[stack_id],
                        thetas_dim=self.hparams.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        dropout=dropout,
                    )
                else:
                    raise ValueError(f"Unknown stack type {stack_type}")

                self.net_blocks.append(net_block)

    def transform_output(
        self,
        y_hat: torch.Tensor,
        target_scale: torch.Tensor | dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Transform output scale."""
        if target_scale is None:
            return y_hat

        if isinstance(target_scale, dict):
            scale = target_scale.get("scale", None)
            center = target_scale.get("center", None)
            if scale is not None and center is not None:
                while scale.dim() < y_hat.dim():
                    scale = scale.unsqueeze(0)
                    center = center.unsqueeze(0)
                return y_hat * scale + center

        scale = target_scale
        while scale.dim() < y_hat.dim():
            scale = scale.unsqueeze(-1)
        return y_hat * scale

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pass forward of network."""
        # target_past is the target history (backcast input)
        target = x.get("target_past", None)
        if target is None:
            # fallback if target_past is not set (e.g. from tests)
            target = x["encoder_cont"][..., 0]

        if target.ndim == 3:
            # if shape is (batch, time, 1), squeeze the last dimension
            target = target.squeeze(-1)

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
            if isinstance(block, NBEATSTrendBlock):
                trend_forecast.append(full)
            elif isinstance(block, NBEATSSeasonalBlock):
                seasonal_forecast.append(full)
            else:
                generic_forecast.append(full)

            # update backcast and forecast (do not use -= inline)
            backcast = backcast - backcast_block
            forecast = forecast + forecast_block

        prediction = forecast
        backcast_val = target - backcast
        trend_val = torch.stack(trend_forecast, dim=0).sum(0)
        seasonal_val = torch.stack(seasonal_forecast, dim=0).sum(0)
        generic_val = torch.stack(generic_forecast, dim=0).sum(0)

        # Scale output back if target_scale is present
        if "target_scale" in x:
            target_scale = x["target_scale"]
            prediction = self.transform_output(prediction, target_scale)
            backcast_val = self.transform_output(backcast_val, target_scale)
            trend_val = self.transform_output(trend_val, target_scale)
            seasonal_val = self.transform_output(seasonal_val, target_scale)
            generic_val = self.transform_output(generic_val, target_scale)

        # Format shapes based on quantiles
        if self.n_quantiles > 1:
            prediction = prediction.unsqueeze(-1).expand(-1, -1, self.n_quantiles)
            backcast_val = backcast_val.unsqueeze(-1).expand(-1, -1, self.n_quantiles)
            trend_val = trend_val.unsqueeze(-1).expand(-1, -1, self.n_quantiles)
            seasonal_val = seasonal_val.unsqueeze(-1).expand(-1, -1, self.n_quantiles)
            generic_val = generic_val.unsqueeze(-1).expand(-1, -1, self.n_quantiles)
        else:
            prediction = prediction.unsqueeze(-1)
            backcast_val = backcast_val.unsqueeze(-1)
            trend_val = trend_val.unsqueeze(-1)
            seasonal_val = seasonal_val.unsqueeze(-1)
            generic_val = generic_val.unsqueeze(-1)

        return {
            "prediction": prediction,
            "backcast": backcast_val,
            "trend": trend_val,
            "seasonality": seasonal_val,
            "generic": generic_val,
        }

    def _compute_combined_loss(
        self,
        y_hat_dict: dict[str, torch.Tensor],
        x: dict[str, torch.Tensor],
        y: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        """Compute the combined loss (forecast loss + optional backcast loss)."""
        y_hat = y_hat_dict["prediction"]
        y_target = y.squeeze(-1) if y.ndim == 3 and y.size(-1) == 1 else y
        loss = self.loss(y_hat, y_target)

        if self.hparams.backcast_loss_ratio > 0:
            backcast = y_hat_dict["backcast"]
            backcast_weight = (
                self.hparams.backcast_loss_ratio
                * self.prediction_length
                / self.context_length
            )
            backcast_weight = backcast_weight / (backcast_weight + 1)
            forecast_weight = 1 - backcast_weight

            target_past = x["target_past"]
            if target_past.ndim == 3 and target_past.size(-1) == 1:
                target_past = target_past.squeeze(-1)

            if isinstance(self.loss, MASE):
                backcast_loss = self.loss(backcast, target_past, y_target)
            else:
                backcast_loss = self.loss(backcast, target_past)

            self.log(
                f"{prefix}_backcast_loss",
                backcast_loss,
                on_step=(prefix == "train"),
                on_epoch=True,
                batch_size=len(y),
            )
            self.log(
                f"{prefix}_forecast_loss",
                loss,
                on_step=(prefix == "train"),
                on_epoch=True,
                batch_size=len(y),
            )
            loss = loss * forecast_weight + backcast_loss * backcast_weight

        return loss

    def training_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """Training step for the model."""
        x, y = batch
        y_hat_dict = self(x)
        loss = self._compute_combined_loss(y_hat_dict, x, y, prefix="train")
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat_dict["prediction"], y, prefix="train")
        return {"loss": loss}

    def validation_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """Validation step for the model."""
        x, y = batch
        y_hat_dict = self(x)
        loss = self._compute_combined_loss(y_hat_dict, x, y, prefix="val")
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat_dict["prediction"], y, prefix="val")
        return {"val_loss": loss}

    def test_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """Test step for the model."""
        x, y = batch
        y_hat_dict = self(x)
        loss = self._compute_combined_loss(y_hat_dict, x, y, prefix="test")
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat_dict["prediction"], y, prefix="test")
        return {"test_loss": loss}

    def plot_interpretation(
        self,
        x: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        idx: int,
        ax=None,
        plot_seasonality_and_generic_on_secondary_axis: bool = False,
    ):
        """Plot decomposition into trend, seasonality and generic forecast."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        else:
            fig = ax[0].get_figure()

        time = torch.arange(-self.context_length, self.prediction_length)

        target_past = x["target_past"]
        if target_past.ndim == 3:
            target_past = target_past[..., 0]
        decoder_target = x.get("decoder_target", None)
        if decoder_target is None:
            # Fallback if decoder_target is not in batch (e.g. at prediction time)
            # Try self.predict or use dummy/zeros if available.
            # During plotting we can pass actual decoder_target.
            decoder_target = torch.zeros(
                (target_past.size(0), self.prediction_length), device=target_past.device
            )

        if decoder_target.ndim == 3:
            decoder_target = decoder_target[..., 0]

        # plot target vs prediction
        ax[0].plot(
            time.cpu(),
            torch.cat([target_past[idx], decoder_target[idx]]).detach().cpu(),
            label="target",
        )

        backcast_pred = output["backcast"][idx]
        if backcast_pred.ndim == 2:
            backcast_pred = backcast_pred[..., 0]
        forecast_pred = output["prediction"][idx]
        if forecast_pred.ndim == 2:
            forecast_pred = forecast_pred[..., 0]

        ax[0].plot(
            time.cpu(),
            torch.cat(
                [
                    backcast_pred.detach(),
                    forecast_pred.detach(),
                ],
                dim=0,
            ).cpu(),
            label="prediction",
        )
        ax[0].set_xlabel("Time")

        # plot blocks
        prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
        next(prop_cycle)  # prediction
        next(prop_cycle)  # observations
        if plot_seasonality_and_generic_on_secondary_axis:
            ax2 = ax[1].twinx()
            ax2.set_ylabel("Seasonality / Generic")
        else:
            ax2 = ax[1]
        for title in ["trend", "seasonality", "generic"]:
            if title not in self.hparams.stack_types:
                continue
            component = output[title][idx]
            if component.ndim == 2:
                component = component[..., 0]
            if title == "trend":
                ax[1].plot(
                    time.cpu(),
                    component.detach().cpu(),
                    label=title.capitalize(),
                    c=next(prop_cycle)["color"],
                )
            else:
                ax2.plot(
                    time.cpu(),
                    component.detach().cpu(),
                    label=title.capitalize(),
                    c=next(prop_cycle)["color"],
                )
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Decomposition")

        fig.legend()
        return fig
