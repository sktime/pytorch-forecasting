"""
N-BEATS v2 model for time series forecasting without covariates.
"""

from typing import Optional
import torch
import torch.nn as nn
from pytorch_forecasting.metrics import MASE

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
    """
    Initialize NBeats Model - use its :py:meth:`~from_dataset` method if possible.

    Based on the article
    `N-BEATS: Neural basis expansion analysis for interpretable time series
        forecasting <http://arxiv.org/abs/1905.10437>`_. The network has (if
    used as ensemble) outperformed all other methods including ensembles of
    traditional statical methods in the M4 competition. The M4 competition is
    arguably the most important benchmark for univariate time series forecasting.

    The :py:class:`~pytorch_forecasting.models.nhits.NHiTS` network has recently
    shown to consistently outperform N-BEATS.

    Parameters
    ----------
    stack_types : list of str
        One of the following values “generic”, “seasonality” or “trend”.
        A list of strings of length 1 or `num_stacks`. Default and recommended
        value for generic mode is ["generic"]. Recommended value for interpretable
        mode is ["trend","seasonality"].
    num_blocks : list of int
        The number of blocks per stack. Length 1 or `num_stacks`. Default for
        generic mode is [1], interpretable mode is [3].
    num_block_layers : list of int
        Number of fully connected layers with ReLU activation per block. Length 1
        or `num_stacks`. Default [4] for both modes.
    width : list of int
        Widths of fully connected layers with ReLU activation. List length 1 or
        `num_stacks`. Default [512] for generic; [256, 2048] for interpretable.
    sharing : list of bool
        Whether weights are shared across blocks in a stack. List length 1 or
        `num_stacks`. Default [False] for generic; [True] for interpretable.
    expansion_coefficient_length : list of int
        If type is "G", length of expansion coefficient; if "T", degree of
        polynomial; if "S", minimum period (e.g., 2 for every timestep). List
        length 1 or `num_stacks`. Default [32] for generic; [3] for interpretable.
    prediction_length : int
        Length of the forecast horizon.
    context_length : int
        Number of time units conditioning the predictions (lookback period).
        Should be between 1-10x `prediction_length`.
    dropout : float
        Dropout probability applied in the network. Helps prevent overfitting.
        Default is 0.1.
    learning_rate : float
        Learning rate used by the optimizer during training. Default is 1e-2.
    log_interval : int
        Interval (in steps) at which training logs are recorded. If -1, logging
        is disabled. Default is -1.
    log_gradient_flow : bool
        Whether to log gradient flow during training. Useful for diagnosing
        vanishing/exploding gradients. Default is False.
    log_val_interval : int
        Interval (in steps) at which validation metrics are logged. If None,
        uses default logging behavior. Default is None.
    weight_decay : float
        Weight decay (L2 regularization) coefficient used by the optimizer to
        reduce overfitting. Default is 1e-3.
    loss
        Loss to optimize. Defaults to `MASE()`.
    reduce_on_plateau_patience : int
        Patience after which learning rate is reduced by factor of 10.
    backcast_loss_ratio : float
        Weight of backcast loss relative to forecast loss. 1.0 gives equal weight;
        default 0.0 means no backcast loss.
    logging_metrics : nn.ModuleList of MultiHorizonMetric
        List of metrics logged during training. Defaults to
        nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()]).
    **kwargs
        Additional arguments forwarded to :py:class:`~BaseModel`.
    """  # noqa: E501


    def __init__(
        self,
        *,
        stack_types: Optional[list[str]] = None,
        num_blocks: Optional[list[int]] = None,
        num_block_layers: Optional[list[int]] = None,
        widths: Optional[list[int]] = None,
        sharing: Optional[list[bool]] = None,
        expansion_coefficient_lengths: Optional[list[int]] = None,
        dropout: float = 0.1,
        backcast_loss_ratio: float = 0.0,
        loss: Optional[Metric] = None,
        logging_metrics: Optional[list[nn.Module]] = None,
        metadata: Optional[dict] = None,
        optimizer: str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
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

        self.stack_types = stack_types or ["trend", "seasonality"]
        self.num_blocks = num_blocks or [3, 3]
        self.num_block_layers = num_block_layers or [3, 3]
        self.widths = widths or [32, 512]
        self.sharing = sharing or [True, True]
        self.expansion_coefficient_lengths = (
            expansion_coefficient_lengths or [3, 7]
        )

        self.dropout = dropout
        self.backcast_loss_ratio = backcast_loss_ratio

        self._init_network()

    def _init_network(self) -> None:
        # N-BEATS is composed of multiple stacks, each containing multiple blocks
        self.net_blocks = nn.ModuleList()

        for stack_id, stack_type in enumerate(self.stack_types):
            for _ in range(self.num_blocks[stack_id]):
                # Select block type based on stack configuration
                if stack_type == "generic":
                    block = NBEATSGenericBlock(
                        units=self.widths[stack_id],
                        thetas_dim=self.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        dropout=self.dropout,
                    )
                elif stack_type == "seasonality":
                    block = NBEATSSeasonalBlock(
                        units=self.widths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        min_period=self.expansion_coefficient_lengths[stack_id],
                        dropout=self.dropout,
                    )
                elif stack_type == "trend":
                    block = NBEATSTrendBlock(
                        units=self.widths[stack_id],
                        thetas_dim=self.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        dropout=self.dropout,
                    )
                else:
                    raise ValueError(f"Unknown stack type: {stack_type}")

                self.net_blocks.append(block)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the N-BEATS model.

        Each block produces a backcast component that explains part of the history
        and a forecast component that contributes to the final prediction.
        Forecasts from all blocks are accumulated, while backcasts are iteratively
        subtracted from the input.
        """
        target = x["target"]
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
            # Each block produces a backcast and forecast component
            backcast_block, forecast_block = block(backcast)

            full = torch.cat(
                [backcast_block.detach(), forecast_block.detach()],
                dim=1,
            )

            if isinstance(block, NBEATSTrendBlock):
                trend_parts.append(full)
            elif isinstance(block, NBEATSSeasonalBlock):
                seasonality_parts.append(full)
            else:
                generic_parts.append(full)

            # Update backcast by removing the explained component
            backcast = backcast - backcast_block
            # Accumulate forecast contributions from all blocks
            forecast = forecast + forecast_block

        prediction = forecast.unsqueeze(-1)

        if "target_scale" in x:
            prediction = self.transform_output(
                prediction, x["target_scale"]
            )

        out = {
            "prediction": prediction,
            "backcast": backcast.unsqueeze(-1),
        }

        if trend_parts:
            out["trend"] = torch.stack(trend_parts).sum(0).unsqueeze(-1)
        if seasonality_parts:
            out["seasonality"] = (
                torch.stack(seasonality_parts).sum(0).unsqueeze(-1)
            )
        if generic_parts:
            out["generic"] = (
                torch.stack(generic_parts).sum(0).unsqueeze(-1)
            )

        return out
    
    def training_step(self, batch, batch_idx):
        """
        Custom training step to preserve v1 N-BEATS behavior.

        Losses such as MASE require access to the encoder target to compute
        scaling factors. The generic v2 BaseModel assumes two-argument losses,
        so this method explicitly passes encoder_target when required.
        """
        x, y = batch
        out = self(x)
        y_hat = out["prediction"]

        if isinstance(self.loss, MASE):
            loss = self.loss(
                y_hat.squeeze(-1),
                y.squeeze(-1),
                x["target"],
            )

        else:
            loss = self.loss(y_hat, y)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss
