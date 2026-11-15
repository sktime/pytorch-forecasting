"""NBEATSx model for the v2 data pipeline."""

from importlib import import_module

import torch
from torch import nn

from pytorch_forecasting.layers._nbeats._nbeatsx import (
    make_nbeatsx_blocks,
    run_nbeatsx_blocks,
)
from pytorch_forecasting.metrics import (
    MASE,
    CrossEntropy,
    DistributionLoss,
    MultiLoss,
    QuantileLoss,
)
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class NBeatsX_v2(BaseModel):
    """NBEATSx for fixed windows and a single continuous target.

    See :class:`~pytorch_forecasting.models.nbeats.NBeatsXForecaster` for model
    parameters. Pass the fitted data pipeline's ``metadata`` when constructing
    this model directly. Forecasts and backcasts use normalized target space.
    """

    @classmethod
    def _pkg(cls):
        """Return the public forecaster without a module-level import cycle."""
        return import_module(
            "pytorch_forecasting.models.nbeats._nbeatsx_forecaster_v2"
        ).NBeatsXForecaster

    def __init__(
        self,
        stack_types: list[str] | None = None,
        num_blocks: list[int] | None = None,
        num_block_layers: list[int] | None = None,
        widths: list[int] | None = None,
        sharing: list[bool] | None = None,
        expansion_coefficient_lengths: list[int] | None = None,
        dropout: float = 0.1,
        backcast_loss_ratio: float = 0.1,
        loss: nn.Module | None = None,
        logging_metrics: list[nn.Module] | None = None,
        optimizer="adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata=None,
    ):
        if metadata is None:
            raise ValueError("NBeatsX_v2 requires encoder-decoder datamodule metadata")
        if metadata["target"] != 1 or metadata.get("target_is_categorical", False):
            raise ValueError("NBeatsX requires a single continuous target")
        self.context_length = metadata["max_encoder_length"]
        self.prediction_length = metadata["max_prediction_length"]
        if (
            metadata["min_encoder_length"] != self.context_length
            or metadata["min_prediction_length"] != self.prediction_length
        ):
            raise ValueError("NBeatsX requires fixed encoder and prediction lengths")
        if isinstance(
            loss,
            (
                CrossEntropy,
                nn.CrossEntropyLoss,
                DistributionLoss,
                QuantileLoss,
                MultiLoss,
            ),
        ):
            raise ValueError("NBeatsX supports single-target point losses only")

        if stack_types is None:
            stack_types = ["generic"]
        super().__init__(
            loss=MASE() if loss is None else loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self.metadata = metadata

        encoder_size = metadata["encoder_cont"] + metadata["encoder_cat"]
        decoder_size = metadata["decoder_cont"] + metadata["decoder_cat"]
        static_size = (
            metadata["static_categorical_features"]
            + metadata["static_continuous_features"]
        )
        if "exogenous" in stack_types and (
            not decoder_size
            or len(metadata.get("known_cont_indices", ())) != metadata["decoder_cont"]
            or len(metadata.get("known_cat_indices", ())) != metadata["decoder_cat"]
        ):
            raise ValueError("Exogenous stacks require known temporal covariates")
        input_size = (
            self.context_length * (1 + encoder_size)
            + self.prediction_length * decoder_size
            + static_size
        )
        self.net_blocks = make_nbeatsx_blocks(
            self.hparams,
            input_size,
            self.context_length,
            self.prediction_length,
            decoder_size,
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Predict from target history, historical features and known future inputs."""
        target = x["target_past"].squeeze(-1)
        encoder = torch.cat([x["encoder_cont"], x["encoder_cat"]], dim=-1)
        decoder = torch.cat([x["decoder_cont"], x["decoder_cat"]], dim=-1)
        static = target.new_empty((target.size(0), 0))
        if "static_continuous_features" in x:
            static = torch.cat(
                [x["static_continuous_features"], x["static_categorical_features"]],
                dim=-1,
            ).flatten(1)

        encoder_basis = None
        if "exogenous" in self.hparams.stack_types:
            encoder_basis = torch.cat(
                [
                    x["encoder_cont"][..., self.metadata["known_cont_indices"]],
                    x["encoder_cat"][..., self.metadata["known_cat_indices"]],
                ],
                dim=-1,
            )
        backcast, block_backcasts, block_forecasts = run_nbeatsx_blocks(
            self.net_blocks, target, encoder, decoder, static, encoder_basis
        )
        return {
            "prediction": block_forecasts.sum(-1).unsqueeze(-1),
            "backcast": backcast.unsqueeze(-1),
            "block_forecasts": block_forecasts.detach(),
            "block_backcasts": block_backcasts.detach(),
        }

    def _step(self, batch, stage):
        x, y = batch
        out = self(x)
        target = x["target_past"].squeeze(-1)
        loss_kwargs = {"encoder_target": target} if isinstance(self._loss, MASE) else {}
        loss = self._loss(out["prediction"], y, **loss_kwargs)
        if self.hparams.backcast_loss_ratio:
            backcast_loss = (target - out["backcast"].squeeze(-1)).square().mean()
            loss = loss + self.hparams.backcast_loss_ratio * backcast_loss
        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=target.size(0),
        )
        self.log_metrics(out["prediction"], y, prefix=stage)
        return loss

    def training_step(self, batch, batch_idx):
        """Compute the training objective."""
        return {"loss": self._step(batch, "train")}

    def validation_step(self, batch, batch_idx):
        """Compute the validation objective."""
        return {"val_loss": self._step(batch, "val")}

    def test_step(self, batch, batch_idx):
        """Compute the test objective."""
        return {"test_loss": self._step(batch, "test")}
