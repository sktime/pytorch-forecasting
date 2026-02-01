"""
N-BEATS v2 model for time series forecasting without covariates.
"""

import torch
import torch.nn as nn

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.layers._nbeats._blocks import (
    NBEATSGenericBlock,
    NBEATSSeasonalBlock,
    NBEATSTrendBlock,
)
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    MASE,
    RMSE,
    SMAPE,
    Metric,
)
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class NBEATS_v2(TslibBaseModel):
    @classmethod
    def _pkg(cls):
        from pytorch_forecasting.models.nbeats._nbeats_v2_pkg import (
            NBEATS_v2_pkg_v2,
        )

        return NBEATS_v2_pkg_v2

    def __init__(
        self,
        *,
        stack_types: list[str] | None = None,
        num_blocks: list[int] | None = None,
        num_block_layers: list[int] | None = None,
        widths: list[int] | None = None,
        sharing: list[bool] | None = None,
        expansion_coefficient_lengths: list[int] | None = None,
        dropout: float = 0.1,
        backcast_loss_ratio: float = 0.0,
        loss: Metric | None = None,
        logging_metrics: list[nn.Module] | None = None,
        metadata: dict | None = None,
        optimizer: str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        context_length: int | None = None,
        prediction_length: int | None = None,
    ):
        if loss is None:
            loss = MASE()

        if logging_metrics is None:
            logging_metrics = [SMAPE(), MAE(), RMSE(), MAPE(), MASE()]

        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            metadata=metadata,
        )
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        if context_length is not None:
            assert (
                context_length == self.metadata["context_length"]
            ), "context_length argument must match metadata['context_length']"

        if prediction_length is not None:
            assert (
                prediction_length == self.metadata["prediction_length"]
            ), "prediction_length argument must match metadata['prediction_length']"

        self.hparams.context_length = self.metadata["context_length"]
        self.hparams.prediction_length = self.metadata["prediction_length"]

        self.context_length = self.hparams.context_length
        self.prediction_length = self.hparams.prediction_length

        self.stack_types = stack_types or ["trend", "seasonality"]
        self.num_blocks = num_blocks or [3, 3]
        self.num_block_layers = num_block_layers or [3, 3]
        self.widths = widths or [32, 512]
        self.sharing = sharing or [True, True]
        self.expansion_coefficient_lengths = expansion_coefficient_lengths or [3, 7]

        self.dropout = dropout
        self.backcast_loss_ratio = backcast_loss_ratio

        self._init_network()

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        assert isinstance(dataset, TimeSeriesDataSet)

        assert isinstance(dataset.target, str), "N-BEATS supports exactly one target."

        assert not isinstance(
            dataset.target_normalizer, NaNLabelEncoder
        ), "Only regression targets are supported."

        assert (
            dataset.min_encoder_length == dataset.max_encoder_length
        ), "Encoder length must be fixed."

        assert (
            dataset.min_prediction_length == dataset.max_prediction_length
        ), "Prediction length must be fixed."

        assert (
            dataset.randomize_length is None
        ), "Randomized sequence length is not supported."

        assert (
            not dataset.add_relative_time_idx
        ), "Relative time index is not supported."

        assert (
            len(dataset.flat_categoricals) == 0
            and len(dataset.reals) == 1
            and dataset._time_varying_unknown_reals == [dataset.target]
        ), "Target must be the only input variable."

        kwargs.setdefault("context_length", dataset.max_encoder_length)
        kwargs.setdefault("prediction_length", dataset.max_prediction_length)

        return cls(**kwargs)

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
        self._stack_blocks = []

        for stack_id, stack_type in enumerate(self.stack_types):
            stack_blocks = nn.ModuleList()

            if self.sharing[stack_id]:
                shared_block = self._make_block(stack_id, stack_type)
                for _ in range(self.num_blocks[stack_id]):
                    stack_blocks.append(shared_block)
            else:
                for _ in range(self.num_blocks[stack_id]):
                    stack_blocks.append(self._make_block(stack_id, stack_type))

            self._stack_blocks.append(stack_blocks)
            self.net_blocks.extend(stack_blocks)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert "history_target" in x
        target = x["history_target"].squeeze(-1)

        assert target.shape[1] == self.context_length

        batch_size = target.size(0)

        forecast = torch.zeros(
            batch_size,
            self.prediction_length,
            device=target.device,
        )

        backcast = target

        trend_parts = []
        seasonality_parts = []
        generic_parts = []

        for block in self.net_blocks:
            backcast_block, forecast_block = block(backcast)

            full = torch.cat([backcast_block.detach(), forecast_block.detach()], dim=1)

            if isinstance(block, NBEATSTrendBlock):
                trend_parts.append(full)
            elif isinstance(block, NBEATSSeasonalBlock):
                seasonality_parts.append(full)
            else:
                generic_parts.append(full)

            backcast = backcast - backcast_block
            forecast = forecast + forecast_block

        prediction = forecast.unsqueeze(-1)
        explained_backcast = (target - backcast).unsqueeze(-1)

        def _empty_component():
            return torch.zeros(
                batch_size,
                self.context_length + self.prediction_length,
                1,
                device=target.device,
            )

        trend = (
            torch.stack(trend_parts).sum(0).unsqueeze(-1)
            if trend_parts
            else _empty_component()
        )
        seasonality = (
            torch.stack(seasonality_parts).sum(0).unsqueeze(-1)
            if seasonality_parts
            else _empty_component()
        )
        generic = (
            torch.stack(generic_parts).sum(0).unsqueeze(-1)
            if generic_parts
            else _empty_component()
        )

        if "target_scale" in x:
            prediction = self.transform_output(prediction, x["target_scale"])
            explained_backcast = self.transform_output(
                explained_backcast, x["target_scale"]
            )
            trend = self.transform_output(trend, x["target_scale"])
            seasonality = self.transform_output(seasonality, x["target_scale"])
            generic = self.transform_output(generic, x["target_scale"])

        return {
            "prediction": prediction,
            "backcast": explained_backcast,
            "trend": trend,
            "seasonality": seasonality,
            "generic": generic,
            "_residual_backcast": backcast,
        }

    def _forecast_loss(self, out, y, x):
        if isinstance(self.loss, MASE):
            return self.loss(
                out["prediction"].squeeze(-1),
                y.squeeze(-1),
                x["history_target"].squeeze(-1),
            )
        return self.loss(out["prediction"], y)

    def _backcast_loss(self, explained_backcast, x):
        backcast = explained_backcast.squeeze(-1)
        target = x["history_target"].squeeze(-1)

        if isinstance(self.loss, MASE):
            return self.loss(backcast, target, target)

        return self.loss(backcast, target)

    def _step(self, batch, stage: str):
        x, y = batch
        out = self(x)

        forecast_loss = self._forecast_loss(out, y, x)
        loss = forecast_loss

        if self.backcast_loss_ratio > 0 and self._supports_backcast_loss():
            weight = (
                self.backcast_loss_ratio * self.prediction_length / self.context_length
            )
            weight = weight / (weight + 1)

            backcast_loss = self._backcast_loss(out["backcast"], x)

            self.log(
                f"{stage}_backcast_loss",
                backcast_loss,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                f"{stage}_forecast_loss",
                forecast_loss,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=False,
            )

            loss = forecast_loss * (1 - weight) + backcast_loss * weight

        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def _supports_backcast_loss(self) -> bool:
        if not isinstance(self.loss, (MASE, MAE, RMSE)):
            return False

        return any(
            isinstance(block, (NBEATSTrendBlock, NBEATSSeasonalBlock))
            for block in self.net_blocks
        )

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, stage="val")
