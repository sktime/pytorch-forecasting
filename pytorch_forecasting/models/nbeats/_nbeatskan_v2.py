"""NBeatsKAN model for PyTorch Forecasting v2."""

from collections.abc import Callable
from typing import Any

from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.layers._nbeats._blocks import (
    NBEATSGenericBlockKAN,
    NBEATSSeasonalBlockKAN,
    NBEATSTrendBlockKAN,
)
from pytorch_forecasting.metrics import MAE, Metric
from pytorch_forecasting.models.nbeats._nbeats_adapter_v2 import NBeatsAdapterV2


class NBeatsKAN_v2(NBeatsAdapterV2):
    """N-BEATS model with Kolmogorov-Arnold Network (KAN) spline layers for v2.

    Parameters
    ----------
    loss : Metric, default=MAE()
        Loss metric used for training and evaluation.
    stack_types : list of str, optional
        List of stack types: ``"generic"``, ``"trend"``, or ``"seasonality"``.
    num_blocks : list of int, optional
        Number of blocks per stack.
    num_block_layers : list of int, optional
        Number of KAN layers per block.
    widths : list of int, optional
        Widths of layers in blocks.
    sharing : list of bool, optional
        Whether blocks share weights per stack.
    expansion_coefficient_lengths : list of int, optional
        Expansion lengths or polynomial degrees per stack.
    dropout : float, default=0.1
        Dropout rate.
    backcast_loss_ratio : float, default=0.0
        Ratio of backcast loss to forecast loss.
    logging_metrics : list of nn.Module, optional
        Logged evaluation metrics.
    optimizer : Optimizer or str, default="adam"
        Optimizer used for training.
    optimizer_params : dict, optional
        Optimizer parameters.
    lr_scheduler : str, optional
        Learning rate scheduler name.
    lr_scheduler_params : dict, optional
        Parameters for the learning rate scheduler.
    num : int, default=5
        KAN grid intervals.
    k : int, default=3
        KAN spline polynomial order.
    noise_scale : float, default=0.5
        KAN noise scale at initialization.
    scale_base_mu : float, default=0.0
        KAN base scale mean.
    scale_base_sigma : float, default=1.0
        KAN base scale std.
    scale_sp : float, default=1.0
        KAN spline scale.
    base_fun : Callable, optional
        KAN residual base activation function.
    grid_eps : float, default=0.02
        KAN grid interpolation parameter.
    grid_range : list of int, optional
        KAN grid range boundaries.
    sp_trainable : bool, default=True
        Whether spline scale is trainable.
    sb_trainable : bool, default=True
        Whether base scale is trainable.
    sparse_init : bool, default=False
        Whether sparse initialization is used.
    metadata : dict, optional
        Metadata from DataModule.
    """

    @classmethod
    def _pkg(cls):
        """Package container for the model."""
        from pytorch_forecasting.models.nbeats._nbeatskan_pkg_v2 import (
            NBeatsKAN_pkg_v2,
        )

        return NBeatsKAN_pkg_v2

    def __init__(
        self,
        loss: Metric = MAE(),
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
        num: int = 5,
        k: int = 3,
        noise_scale: float = 0.5,
        scale_base_mu: float = 0.0,
        scale_base_sigma: float = 1.0,
        scale_sp: float = 1.0,
        base_fun: Callable | None = None,
        grid_eps: float = 0.02,
        grid_range: list[int] | None = None,
        sp_trainable: bool = True,
        sb_trainable: bool = True,
        sparse_init: bool = False,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ):
        if base_fun is None:
            base_fun = nn.SiLU()
        if grid_range is None:
            grid_range = [-1, 1]
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
            metadata=metadata,
            backcast_loss_ratio=backcast_loss_ratio,
        )
        self.save_hyperparameters(
            ignore=["loss", "logging_metrics", "metadata", "base_fun"]
        )

        self.stack_types = stack_types
        self.num_blocks = num_blocks
        self.num_block_layers = num_block_layers
        self.widths = widths
        self.sharing = sharing
        self.expansion_coefficient_lengths = expansion_coefficient_lengths
        self.dropout = dropout

        self.kan_params = {
            "num": num,
            "k": k,
            "noise_scale": noise_scale,
            "scale_base_mu": scale_base_mu,
            "scale_base_sigma": scale_base_sigma,
            "scale_sp": scale_sp,
            "base_fun": base_fun,
            "grid_eps": grid_eps,
            "grid_range": grid_range,
            "sp_trainable": sp_trainable,
            "sb_trainable": sb_trainable,
            "sparse_init": sparse_init,
        }

        self._init_network()

    def _init_network(self):
        """Build N-BEATS KAN stacks."""
        self.net_blocks = nn.ModuleList()
        for stack_id, stack_type in enumerate(self.stack_types):
            for _ in range(self.num_blocks[stack_id]):
                net_block: nn.Module
                if stack_type == "generic":
                    net_block = NBEATSGenericBlockKAN(
                        units=self.widths[stack_id],
                        thetas_dim=self.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        dropout=self.dropout,
                        **self.kan_params,
                    )
                elif stack_type == "seasonality":
                    net_block = NBEATSSeasonalBlockKAN(
                        units=self.widths[stack_id],
                        thetas_dim=self.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        nb_harmonics=None,  # type: ignore[arg-type]
                        min_period=self.expansion_coefficient_lengths[stack_id],
                        dropout=self.dropout,
                        **self.kan_params,
                    )
                elif stack_type == "trend":
                    net_block = NBEATSTrendBlockKAN(
                        units=self.widths[stack_id],
                        thetas_dim=self.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        dropout=self.dropout,
                        **self.kan_params,
                    )
                else:
                    raise ValueError(f"Unknown stack_type: {stack_type}")

                self.net_blocks.append(net_block)
