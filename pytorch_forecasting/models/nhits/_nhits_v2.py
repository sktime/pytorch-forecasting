"""Native v2 N-HiTS model implementation."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel
from pytorch_forecasting.models.nhits.sub_modules import NHiTS as NHiTSModule


class NHiTS(TslibBaseModel):
    """N-HiTS model implemented on top of the v2 `TslibBaseModel` API."""

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.nhits._nhits_pkg_v2 import NHiTS_pkg_v2

        return NHiTS_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        static_hidden_size: int | None = None,
        naive_level: bool = True,
        shared_weights: bool = True,
        activation: str = "ReLU",
        initialization: str = "lecun_normal",
        n_blocks: list[int] | None = None,
        n_layers: int | list[int] = 2,
        hidden_size: int = 512,
        pooling_sizes: list[int] | None = None,
        downsample_frequencies: list[int] | None = None,
        pooling_mode: str = "max",
        interpolation_mode: str = "linear",
        batch_normalization: bool = False,
        dropout: float = 0.0,
        backcast_loss_ratio: float = 0.0,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
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

        if n_blocks is None:
            n_blocks = [1, 1, 1]

        self.hidden_size = hidden_size
        self.backcast_loss_ratio = backcast_loss_ratio
        self.n_stacks = len(n_blocks)

        if pooling_sizes is None:
            if self.prediction_length <= 1:
                pooling_sizes = [1] * self.n_stacks
            else:
                pooling_sizes = np.exp2(
                    np.round(
                        np.linspace(
                            0.49, np.log2(self.prediction_length / 2), self.n_stacks
                        )
                    )
                )
                pooling_sizes = [max(int(x), 1) for x in pooling_sizes[::-1]]

        if downsample_frequencies is None:
            downsample_frequencies = [
                max(min(self.prediction_length, int(np.power(x, 1.5))), 1)
                for x in pooling_sizes
            ]

        if static_hidden_size is None:
            static_hidden_size = hidden_size

        if isinstance(n_layers, int):
            n_layers = [n_layers] * self.n_stacks

        if len(n_layers) != self.n_stacks:
            raise ValueError("Length of n_layers must match length of n_blocks.")

        if len(pooling_sizes) != self.n_stacks:
            raise ValueError("Length of pooling_sizes must match length of n_blocks.")

        if len(downsample_frequencies) != self.n_stacks:
            raise ValueError(
                "Length of downsample_frequencies must match length of n_blocks."
            )

        self.n_quantiles = None
        if isinstance(self.loss, QuantileLoss):
            self.n_quantiles = len(self.loss.quantiles)

        self.output_size = [self._per_target_output_size] * self.target_dim

        self.encoder_covariate_size = self.cont_dim + self.cat_dim

        known_features = set(self.metadata.get("feature_names", {}).get("known", []))
        continuous_features = set(
            self.metadata.get("feature_names", {}).get("continuous", [])
        )
        categorical_features = set(
            self.metadata.get("feature_names", {}).get("categorical", [])
        )

        self.decoder_covariate_size = len(
            known_features.intersection(continuous_features)
        ) + len(known_features.intersection(categorical_features))

        self.static_size = self.static_cat_dim + self.static_cont_dim

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self.model = NHiTSModule(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            output_size=self.output_size,
            static_size=self.static_size,
            encoder_covariate_size=self.encoder_covariate_size,
            decoder_covariate_size=self.decoder_covariate_size,
            static_hidden_size=static_hidden_size,
            n_blocks=n_blocks,
            n_layers=n_layers,
            hidden_size=self.n_stacks * [2 * [self.hidden_size]],
            pooling_sizes=pooling_sizes,
            downsample_frequencies=downsample_frequencies,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            dropout=dropout,
            activation=activation,
            initialization=initialization,
            batch_normalization=batch_normalization,
            shared_weights=shared_weights,
            naive_level=naive_level,
        )

    @property
    def _per_target_output_size(self) -> int:
        return self.n_quantiles if self.n_quantiles is not None else 1

    def _combine_features(
        self, x: dict[str, torch.Tensor], keys: tuple[str, ...]
    ) -> torch.Tensor | None:
        parts = []
        for key in keys:
            value = x.get(key)
            if value is not None and value.size(-1) > 0:
                parts.append(value.float())

        if not parts:
            return None

        return torch.cat(parts, dim=-1)

    def _prepare_static_features(
        self, x: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        static_features = x.get("static_continuous_features")
        if static_features is None:
            static_features = x.get("static_categorical_features")

        if static_features is None:
            return None

        if static_features.ndim == 3:
            static_features = static_features[:, 0, :]

        if static_features.size(-1) == 0:
            return None

        return static_features.float()

    def _reshape_forecast(self, forecast: torch.Tensor) -> torch.Tensor:
        batch_size, pred_len, _ = forecast.shape

        if self.n_quantiles is not None:
            forecast = forecast.reshape(
                batch_size, pred_len, self.target_dim, self.n_quantiles
            )
            if self.target_dim == 1:
                forecast = forecast.squeeze(2)
        else:
            forecast = forecast.reshape(batch_size, pred_len, self.target_dim)

        return forecast

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run N-HiTS forward pass for v2 datamodule batches."""
        encoder_y = x["history_target"].float()
        if encoder_y.ndim == 2:
            encoder_y = encoder_y.unsqueeze(-1)

        encoder_mask = x.get("history_mask")
        if encoder_mask is None:
            encoder_mask = torch.ones(
                encoder_y.shape[:2], dtype=torch.bool, device=encoder_y.device
            )

        encoder_x_t = self._combine_features(x, ("history_cont", "history_cat"))
        decoder_x_t = self._combine_features(x, ("future_cont", "future_cat"))
        x_s = self._prepare_static_features(x)

        forecast, backcast, block_forecasts, block_backcasts = self.model(
            encoder_y=encoder_y,
            encoder_mask=encoder_mask,
            encoder_x_t=encoder_x_t,
            decoder_x_t=decoder_x_t,
            x_s=x_s,
        )

        prediction = self._reshape_forecast(forecast)

        target_scale = x.get("target_scale")
        if isinstance(target_scale, dict) and {"scale", "center"}.issubset(
            target_scale.keys()
        ):
            prediction = self.transform_output(prediction, target_scale)

        return {
            "prediction": prediction,
            "backcast": backcast,
            "block_forecasts": block_forecasts,
            "block_backcasts": block_backcasts,
        }

    def step(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
        stage: str,
    ) -> dict[str, torch.Tensor]:
        """Unified v2 step for train/val/test loops."""
        x, y = batch
        output = self(x)

        prediction = output["prediction"]
        loss = self.loss(prediction, y)

        if self.backcast_loss_ratio > 0:
            backcast = output["backcast"]
            history_target = x["history_target"]

            if backcast.ndim == 3 and backcast.size(-1) == 1:
                backcast = backcast[..., 0]
            if history_target.ndim == 3 and history_target.size(-1) == 1:
                history_target = history_target[..., 0]

            backcast_loss = self.loss(backcast, history_target)
            loss = (
                1 - self.backcast_loss_ratio
            ) * loss + self.backcast_loss_ratio * backcast_loss

            self.log(
                f"{stage}_backcast_loss",
                backcast_loss,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log_metrics(prediction, y, prefix=stage)

        if stage == "train":
            return {"loss": loss}
        if stage == "val":
            return {"val_loss": loss}
        return {"test_loss": loss}

    def training_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        return self.step(batch, batch_idx, stage="train")

    def validation_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        return self.step(batch, batch_idx, stage="val")

    def test_step(
        self, batch: tuple[dict[str, torch.Tensor], torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        return self.step(batch, batch_idx, stage="test")
