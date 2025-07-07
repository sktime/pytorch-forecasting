"""
N-Beats model for timeseries forecasting without covariates.
"""

from typing import Optional

from torch import nn

from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric
from pytorch_forecasting.models.nbeats._nbeats_adapter import NBeatsAdapter
from pytorch_forecasting.models.nbeats.sub_modules import (
    NBEATSGenericBlock,
    NBEATSSeasonalBlock,
    NBEATSTrendBlock,
)


class NBeats(NBeatsAdapter):
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
        stack_types: Optional[list[str]] = None,
        num_blocks: Optional[list[int]] = None,
        num_block_layers: Optional[list[int]] = None,
        widths: Optional[list[int]] = None,
        sharing: Optional[list[bool]] = None,
        expansion_coefficient_lengths: Optional[list[int]] = None,
        prediction_length: int = 1,
        context_length: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-2,
        log_interval: int = -1,
        log_gradient_flow: bool = False,
        log_val_interval: int = None,
        weight_decay: float = 1e-3,
        loss: MultiHorizonMetric = None,
        reduce_on_plateau_patience: int = 1000,
        backcast_loss_ratio: float = 0.0,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
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
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        if loss is None:
            loss = MASE()

        self.save_hyperparameters(ignore=["loss", "logging_metrics"])
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)
        # setup stacks
        self.net_blocks = nn.ModuleList()
        for stack_id, stack_type in enumerate(stack_types):
            for _ in range(num_blocks[stack_id]):
                if stack_type == "generic":
                    net_block = NBEATSGenericBlock(
                        units=self.hparams.widths[stack_id],
                        thetas_dim=self.hparams.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        dropout=dropout,
                    )
                elif stack_type == "seasonality":
                    net_block = NBEATSSeasonalBlock(
                        units=self.hparams.widths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        min_period=expansion_coefficient_lengths[stack_id],
                        dropout=dropout,
                    )
                elif stack_type == "trend":
                    net_block = NBEATSTrendBlock(
                        units=self.hparams.widths[stack_id],
                        thetas_dim=self.hparams.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        dropout=dropout,
                    )
                else:
                    raise ValueError(f"Unknown stack type {stack_type}")

                self.net_blocks.append(net_block)
