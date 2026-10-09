"""
N-Beats model adapter for timeseries forecasting (v2).
"""

from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.layers._nbeats._blocks import (
    NBEATSSeasonalBlock,
    NBEATSTrendBlock,
)
from pytorch_forecasting.metrics import Metric
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class NBeatsAdapterV2(BaseModel):
    """
    Shared forward and training logic for the N-Beats model family (v2).

    Subclasses define stack construction in ``_init_network``; this
    adapter implements the iterative backcast/forecast loop and optional
    backcast loss.

    Univariate models use ``target_past``; exogenous variants (e.g. NBEATx) can
    extend ``forward`` to consume ``encoder_cont`` / ``decoder_cont``.

    Parameters
    ----------
    loss : Metric
        Loss function for the forecast horizon.
    logging_metrics : list[nn.Module], optional
        Metrics to log during training, validation, and testing.
    optimizer : Optimizer or str, optional
        Optimizer for training. Default is ``"adam"``.
    optimizer_params : dict, optional
        Keyword arguments passed to the optimizer constructor.
    lr_scheduler : str, optional
        Learning rate scheduler name.
    lr_scheduler_params : dict, optional
        Keyword arguments passed to the scheduler constructor.
    metadata : dict, optional
        Metadata from ``EncoderDecoderTimeSeriesDataModule`` (``max_encoder_length``,
        ``max_prediction_length``, ``encoder_cont``, etc.).
    backcast_loss_ratio : float, default=0.0
        Weight of the backcast reconstruction term relative to forecast loss.
        When ``0``, only forecast loss is used. When positive, train, validation,
        and test steps combine forecast and backcast losses.
    **kwargs
        Ignored; reserved for subclass hyperparameters.
    """

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
        )
        self.metadata = metadata or {}
        self.context_length = self.metadata.get("max_encoder_length", 0)
        self.prediction_length = self.metadata.get("max_prediction_length", 0)
        self.encoder_cont_dim = self.metadata.get("encoder_cont", 0)
        self.decoder_cont_dim = self.metadata.get("decoder_cont", 0)
        self.backcast_loss_ratio = backcast_loss_ratio

    def _target_from_batch(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract univariate target history from an encoder-decoder batch.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input batch. Uses ``target_past`` (v2 encoder-decoder) or, as a
            fallback, ``history_target`` (tslib batches).

        Returns
        -------
        torch.Tensor
            Target history of shape ``(batch_size, context_length)``.
        """
        if "target_past" in x:
            target = x["target_past"]
        elif "history_target" in x:
            target = x["history_target"]
        else:
            raise KeyError("Batch must contain 'target_past' or 'history_target'.")

        if target.ndim == 3:
            target = target[..., 0]
        return target

    def transform_output(
        self,
        y_hat: torch.Tensor,
        target_scale: torch.Tensor | dict[str, torch.Tensor] | list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Rescale model outputs to the original target scale.

        Parameters
        ----------
        y_hat : torch.Tensor
            Normalized model output.
        target_scale : torch.Tensor or dict
            Scale information from the batch. Encoder-decoder batches provide a
            tensor; tslib batches may provide a dict with ``scale`` and ``center``.

        Returns
        -------
        torch.Tensor
            Output rescaled to the original target scale.
        """
        if isinstance(target_scale, dict):
            scale = target_scale["scale"]
            center = target_scale.get("center", 0)
            while scale.dim() < y_hat.dim():
                scale = scale.unsqueeze(-1)
                if torch.is_tensor(center):
                    center = center.unsqueeze(-1)
            return y_hat * scale + center

        scale = (
            target_scale[0] if isinstance(target_scale, (list, tuple)) else target_scale
        )
        while scale.dim() < y_hat.dim():
            scale = scale.unsqueeze(-1)
        return y_hat * scale

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass through the N-Beats block stack.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input batch from the datamodule. Must contain ``target_past`` (or
            ``history_target``). May contain ``target_scale`` for inverse scaling.

        Returns
        -------
        dict[str, torch.Tensor]
            Model outputs with keys ``prediction``, ``backcast``, ``trend``,
            ``seasonality``, and ``generic``.
        """
        target = self._target_from_batch(x)

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

        backcast = target
        for block in self.net_blocks:
            backcast_block, forecast_block = block(backcast)

            full = torch.cat([backcast_block.detach(), forecast_block.detach()], dim=1)
            if isinstance(block, NBEATSTrendBlock):
                trend_forecast.append(full)
            elif isinstance(block, NBEATSSeasonalBlock):
                seasonal_forecast.append(full)
            else:
                generic_forecast.append(full)

            backcast = (
                backcast - backcast_block
            )  # do not use backcast -= backcast_block as this signifies an inline operation  # noqa: E501
            forecast = forecast + forecast_block

        prediction = forecast.unsqueeze(-1)
        backcast_out = (target - backcast).unsqueeze(-1)
        trend = torch.stack(trend_forecast, dim=0).sum(0).unsqueeze(-1)
        seasonality = torch.stack(seasonal_forecast, dim=0).sum(0).unsqueeze(-1)
        generic = torch.stack(generic_forecast, dim=0).sum(0).unsqueeze(-1)

        if "target_scale" in x:
            prediction = self.transform_output(prediction, x["target_scale"])
            backcast_out = self.transform_output(backcast_out, x["target_scale"])
            trend = self.transform_output(trend, x["target_scale"])
            seasonality = self.transform_output(seasonality, x["target_scale"])
            generic = self.transform_output(generic, x["target_scale"])

        return {
            "prediction": prediction,
            "backcast": backcast_out,
            "trend": trend,
            "seasonality": seasonality,
            "generic": generic,
        }

    def _compute_loss(
        self,
        x: dict[str, torch.Tensor],
        y: torch.Tensor,
        out: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute forecast loss, optionally combined with backcast loss.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input batch (used for encoder target when backcast loss is enabled).
        y : torch.Tensor
            Forecast horizon target.
        out : dict[str, torch.Tensor]
            Forward pass output.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss for logging and optimization.
        y_hat : torch.Tensor
            Forecast predictions from ``out["prediction"]``.
        """
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

            backcast_loss = (backcast - encoder_target).abs().mean() * backcast_weight
            loss = loss * forecast_weight + backcast_loss

        return loss, y_hat

    def training_step(
        self, batch: tuple[dict[str, torch.Tensor]], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Training step with optional backcast loss.

        Parameters
        ----------
        batch : tuple[dict[str, torch.Tensor]]
            ``(x, y)`` from the dataloader.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with key ``loss``.
        """
        x, y = batch
        out = self(x)
        loss, y_hat = self._compute_loss(x, y, out)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat, y, prefix="train")
        return {"loss": loss}

    def validation_step(
        self, batch: tuple[dict[str, torch.Tensor]], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Validation step with optional backcast loss.

        Parameters
        ----------
        batch : tuple[dict[str, torch.Tensor]]
            ``(x, y)`` from the dataloader.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with key ``val_loss``.
        """
        x, y = batch
        out = self(x)
        loss, y_hat = self._compute_loss(x, y, out)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat, y, prefix="val")
        return {"val_loss": loss}

    def test_step(
        self, batch: tuple[dict[str, torch.Tensor]], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Test step with optional backcast loss.

        Parameters
        ----------
        batch : tuple[dict[str, torch.Tensor]]
            ``(x, y)`` from the dataloader.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with key ``test_loss``.
        """
        x, y = batch
        out = self(x)
        loss, y_hat = self._compute_loss(x, y, out)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat, y, prefix="test")
        return {"test_loss": loss}
