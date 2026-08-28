"""
N-Beats model for pytorch-forecasting v2 (no covariates).
"""

from typing import Any

from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.layers._nbeats._blocks import (
    NBEATSGenericBlock,
    NBEATSSeasonalBlock,
    NBEATSTrendBlock,
)
from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE, Metric
from pytorch_forecasting.models.nbeats._nbeats_adapter_v2 import NBeatsAdapterV2


class NBeats(NBeatsAdapterV2):
    """
    N-BEATS for pytorch-forecasting v2.

    Based on
    `N-BEATS: Neural basis expansion analysis for interpretable time series
    forecasting <http://arxiv.org/abs/1905.10437>`_.

    Network construction matches the v1 ``NBeats`` class; ``context_length`` /
    ``prediction_length`` come from datamodule ``metadata`` instead of
    ``from_dataset``.
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
        if logging_metrics is None:
            logging_metrics = [SMAPE(), MAE(), RMSE(), MAPE()]

        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            metadata=metadata,
            backcast_loss_ratio=backcast_loss_ratio,
        )
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self.stack_types = stack_types
        self.num_blocks = num_blocks
        self.num_block_layers = num_block_layers
        self.widths = widths
        self.sharing = sharing
        self.expansion_coefficient_lengths = expansion_coefficient_lengths
        self.dropout = dropout

        self._init_network()

    def _init_network(self):
        """Build N-BEATS stacks (same block wiring as v1)."""
        self.net_blocks = nn.ModuleList()
        for stack_id, stack_type in enumerate(self.stack_types):
            for _ in range(self.num_blocks[stack_id]):
                if stack_type == "generic":
                    net_block = NBEATSGenericBlock(
                        units=self.widths[stack_id],
                        thetas_dim=self.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        dropout=self.dropout,
                    )
                elif stack_type == "seasonality":
                    net_block = NBEATSSeasonalBlock(
                        units=self.widths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        min_period=self.expansion_coefficient_lengths[stack_id],
                        dropout=self.dropout,
                    )
                elif stack_type == "trend":
                    net_block = NBEATSTrendBlock(
                        units=self.widths[stack_id],
                        thetas_dim=self.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        dropout=self.dropout,
                    )
                else:
                    raise ValueError(f"Unknown stack type {stack_type}")

                self.net_blocks.append(net_block)


NBeats_v2 = NBeats
