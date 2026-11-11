"""
N-Beats model for pytorch-forecasting v2 (no covariates).
"""

from typing import Any, Optional, Union

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
    N-BEATS for pytorch-forecasting v2 (univariate time series forecasting).

    Based on the article `N-BEATS: Neural basis expansion analysis for
    interpretable time series forecasting
    <http://arxiv.org/abs/1905.10437>`_. The network has (if used as
    ensemble) outperformed all other methods including ensembles of
    traditional statistical methods in the M4 competition. The M4
    competition is arguably the most important benchmark for univariate
    time series forecasting.

    Parameters
    ----------
    loss : Metric
        Loss metric to optimize during training.
    stack_types : list of str, optional
        One of "generic", "seasonality", or "trend". A list of strings
        of length equal to the number of stacks. Default is ["trend",
        "seasonality"] for interpretable mode.
    num_blocks : list of int, optional
        The number of blocks per stack. List length equal to number of
        stacks. Default is [3, 3].
    num_block_layers : list of int, optional
        Number of fully connected layers with ReLU activation per block.
        List length equal to number of stacks. Default is [3, 3].
    widths : list of int, optional
        Widths of fully connected layers with ReLU activation. List
        length equal to number of stacks. Default is [32, 512].
    sharing : list of bool, optional
        Whether weights are shared across blocks within a stack. List
        length equal to number of stacks. Default is [True, True].
    expansion_coefficient_lengths : list of int, optional
        If type is "generic", length of expansion coefficients; if
        "trend", degree of polynomial; if "seasonality", minimum period.
        List length equal to number of stacks. Default is [3, 7].
    dropout : float, optional
        Dropout probability applied in the network. Helps prevent
        overfitting. Default is 0.1.
    backcast_loss_ratio : float, optional
        Weight of backcast loss relative to forecast loss. 1.0 gives
        equal weight; 0.0 means no backcast loss. Default is 0.0.
    logging_metrics : list of nn.Module, optional
        List of metrics logged during training. Defaults to
        [SMAPE(), MAE(), RMSE(), MAPE()].
    optimizer : Optimizer or str, optional
        Optimizer to use for training. Can be a torch.optim.Optimizer
        class or string like "adam". Default is "adam".
    optimizer_params : dict, optional
        Additional parameters for the optimizer. Default is None.
    lr_scheduler : str, optional
        Learning rate scheduler type. Default is None.
    lr_scheduler_params : dict, optional
        Additional parameters for the learning rate scheduler. Default
        is None.
    metadata : dict, optional
        Additional metadata for the model. Default is None.
    **kwargs
        Additional arguments forwarded to :py:class:`~NBeatsAdapterV2`.
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
