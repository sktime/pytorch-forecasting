"""
N-BEATS v2 model for time series forecasting without covariates.
"""

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
    """
    Initialize NBEATS Model.

    Based on the article
    `N-BEATS: Neural basis expansion analysis for interpretable time series
        forecasting <http://arxiv.org/abs/1905.10437>`_. The network uses stacks
    of fully connected blocks to model time series and decomposes the signal into
    interpretable components such as trend and seasonality through backward and
    forward residual learning.

    Parameters
    ----------
    stack_types : list of str
        One of the following values “generic”, “seasonality” or “trend”.
        A list of strings of length equal to the number of stacks.
        Default is ["trend", "seasonality"].
    num_blocks : list of int
        The number of blocks per stack. Length should match `stack_types`.
        Default is [3, 3].
    num_block_layers : list of int
        Number of fully connected layers with ReLU activation per block.
        Length should match `stack_types`. Default is [3, 3].
    widths : list of int
        Widths of fully connected layers with ReLU activation.
        Length should match `stack_types`. Default is [32, 512].
    sharing : list of bool
        Whether weights are shared across blocks in a stack.
        Length should match `stack_types`. Default is [True, True].
    expansion_coefficient_lengths : list of int
        If type is "generic", length of expansion coefficient;
        if "trend", degree of polynomial; if "seasonality",
        minimum period. Length should match `stack_types`.
        Default is [3, 7].
    dropout : float
        Dropout probability applied in the network. Default is 0.0.
    loss
        Loss to optimize. Defaults to `MASE()`.
    logging_metrics : list of nn.Module
        Metrics logged during training.
    optimizer : Optimizer or str
        Optimizer used for training. Default is "adam".
    optimizer_params : dict
        Parameters passed to the optimizer.
    lr_scheduler : str
        Learning rate scheduler name.
    lr_scheduler_params : dict
        Parameters passed to the learning rate scheduler.
    metadata : dict
        Dictionary containing dataset metadata. Must include
        ``max_encoder_length`` and ``max_prediction_length``.
    **kwargs
        Additional arguments forwarded to :class:`~BaseModel`.
    """

    @classmethod
    def _pkg(cls):
        """Package for the model."""
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
        """Initialize the NBEATS_v2 model."""
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
        """Create a single N-BEATS block for the given stack."""
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
        """Initialize N-BEATS stacks and blocks."""
        self.net_blocks = nn.ModuleList()

        for stack_id, stack_type in enumerate(self.stack_types):
            if self.sharing[stack_id]:
                block = self._make_block(stack_id, stack_type)
                for _ in range(self.num_blocks[stack_id]):
                    self.net_blocks.append(block)
            else:
                for _ in range(self.num_blocks[stack_id]):
                    self.net_blocks.append(self._make_block(stack_id, stack_type))

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the N-BEATS v2 model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input batch containing past target values.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with forecast and decomposition components.
        """
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
