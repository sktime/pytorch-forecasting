"""
N-Beats model with KAN blocks for timeseries forecasting without covariates.
"""

from typing import Optional

import torch
from torch import nn

from pytorch_forecasting.layers._nbeats._blocks import (
    NBEATSGenericBlockKAN,
    NBEATSSeasonalBlockKAN,
    NBEATSTrendBlockKAN,
)
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric
from pytorch_forecasting.models.nbeats._nbeats_adapter import NBeatsAdapter


class NBeatsKAN(NBeatsAdapter):
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
        One of the following values: “generic”, “seasonality" or
        “trend". A list of strings of length 1 or 'num_stacks'. Default and
        recommended value for generic mode: [“generic”] Recommended value for
        interpretable mode: [“trend”,”seasonality”].
    num_blocks : list of int
        The number of blocks per stack. A list of ints of length 1 or
        'num_stacks'. Default and recommended value for generic mode: [1]
        Recommended value for interpretable mode: [3]
    num_block_layers : list of int
        Number of fully connected layers with ReLu activation per block.
        A list of ints of length 1 or 'num_stacks'. Default and recommended
        value for generic mode: [4] Recommended value for interpretable mode:
        [4].
    widths : list of int
        Widths of the fully connected layers with ReLu activation in the
        blocks. A list of ints of length 1 or 'num_stacks'. Default and
        recommended value for generic mode: [512]. Recommended value for
        interpretable mode: [256, 2048]
    sharing : list of bool
        Whether the weights are shared with the other blocks per stack.
        A list of ints of length 1 or 'num_stacks'. Default and recommended
        value for generic mode: [False]. Recommended value for interpretable
        mode: [True].
    expansion_coefficient_lengths : list of int
        If the type is “G” (generic), then the length of the expansion coefficient.
        If type is “T” (trend), then it corresponds to the degree of the
        polynomial.
        If the type is “S” (seasonal) then this is the minimum period allowed,
        e.g. 2 for changes every timestep. A list of ints of length 1 or
        'num_stacks'. Default value for generic mode: [32] Recommended value for
        interpretable mode: [3]
    prediction_length : int
        Length of the prediction. Also known as 'horizon'.
    context_length : int
        Number of time units that condition the predictions.
        Also known as 'lookback period'.
        Should be between 1-10 times the prediction length.
    backcast_loss_ratio : float
        Weight of backcast in comparison to forecast when calculating the loss.
        A weight of 1.0 means that forecast and backcast loss is weighted the same
        (regardless of backcast and forecast lengths). Defaults to 0.0, i.e. no weight.
    loss : MultiHorizonMetric
        Loss to optimize. Defaults to MASE().
    log_gradient_flow : bool
        If to log gradient flow, this takes time and should be only done to diagnose
        training failures.
    reduce_on_plateau_patience : int
        Patience after which learning rate is reduced by a factor of 10
    logging_metrics : nn.ModuleList of MultiHorizonMetric
        List of metrics that are logged during training. Defaults to
        nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
    num : int
        Parameter for KAN layer. the number of grid intervals = G.
        Default: 5.
    k : int
        Parameter for KAN layer. the order of piecewise polynomial. Default: 3.
    noise_scale : float
        Parameter for KAN layer. the scale of noise injected at initialization.
        Default: 0.1.
    scale_base_mu : float
        Parameter for KAN layer. the scale of the residual function b(x) is initialized
        to be N(scale_base_mu, scale_base_sigma^2). Default: 0.0.
    scale_base_sigma : float
        Parameter for KAN layer. the scale of the residual function b(x) is initialized
        to be N(scale_base_mu, scale_base_sigma^2). Default: 1.0.
    scale_sp : float
        Parameter for KAN layer. the scale of the base function spline(x). Default: 1.0.
    base_fun : callable
        Parameter for KAN layer. residual function b(x). Default: None.
    grid_eps : float
        Parameter for KAN layer. When grid_eps = 1, the grid is uniform;
        when grid_eps = 0, the grid is partitioned using percentiles of samples.
        0 < grid_eps < 1 interpolates between the two extremes. Default: 0.02.
    grid_range : list of int
        Parameter for KAN layer. list/np.array of shape (2,). setting the range of grids.
        Default: None.
    sp_trainable : bool
        Parameter for KAN layer. If true, scale_sp is trainable. Default: True.
    sb_trainable : bool
        Parameter for KAN layer. If true, scale_base is trainable. Default: True.
    sparse_init : bool
        Parameter for KAN layer. if sparse_init = True, sparse initialization is applied.
        Default: False.
    **kwargs
        Additional arguments to :py:class:`~BaseModel`.

    Examples
    --------
    See the full example in:
    `examples/nbeats_with_kan.py`

    Notes
    --------
    The KAN blocks are based on the Kolmogorov-Arnold representation theorem and replace fixed MLP edge weights
    with learnable univariate spline functions. This allows KAN-augmented N-BEATS to better capture complex patterns,
    improve interpretability, and achieve parameter efficiency. Additionally, when applied in a doubly-residual
    adversarial framework, the model excels at zero-shot time-series forecasting across markets.

    Key differences from original N-BEATS:
    - MLP layers are replaced by KAN layers with spline-based edge functions.
    - Each weight is a trainable function, not a scalar.
    - Enables visualization of learned functions and better domain adaptation.
    - Yields improved accuracy and interpretability with fewer parameters.

    References
    ----------
    .. [1] Z. Liu et al. (2024), “KAN: Kolmogorov-Arnold Networks”
    propose replacing MLP weights with spline-based learnable edge functions, enabling improved accuracy,
    interpretability, and scaling behavior compared to standard MLPs.
    .. [2] A. Bhattacharya & N. Haq (2024), “Zero Shot Time Series Forecasting Using Kolmogorov Arnold Networks”
    incorporate KAN layers into a doubly-residual N-BEATS architecture with adversarial domain adaptation,
    achieving strong zero-shot cross-market electricity price forecasting performance.
    """  # noqa: E501

    @classmethod
    def _pkg(cls):
        """Package for the model."""
        from pytorch_forecasting.models.nbeats._nbeatskan_pkg import NBeatsKAN_pkg

        return NBeatsKAN_pkg

    def __init__(
        self,
        stack_types: list[str] | None = None,
        num_blocks: list[int] | None = None,
        num_block_layers: list[int] | None = None,
        widths: list[int] | None = None,
        sharing: list[bool] | None = None,
        expansion_coefficient_lengths: list[int] | None = None,
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
        num: int = 5,
        k: int = 3,
        noise_scale: float = 0.5,
        scale_base_mu: float = 0.0,
        scale_base_sigma: float = 1.0,
        scale_sp: float = 1.0,
        base_fun: callable = None,
        grid_eps: float = 0.02,
        grid_range: list[int] = None,
        sp_trainable: bool = True,
        sb_trainable: bool = True,
        sparse_init: bool = False,
        **kwargs,
    ):
        if base_fun is None:
            base_fun = torch.nn.SiLU()
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
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        if loss is None:
            loss = MASE()

        self.save_hyperparameters(ignore=["loss", "logging_metrics"])
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        # Bundle KAN parameters into a dictionary
        kan_params = {
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
        self.kan_params = kan_params
        # setup stacks
        self.net_blocks = nn.ModuleList()
        for stack_id, stack_type in enumerate(stack_types):
            for _ in range(num_blocks[stack_id]):
                if stack_type == "generic":
                    net_block = NBEATSGenericBlockKAN(
                        units=self.hparams.widths[stack_id],
                        thetas_dim=self.hparams.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        dropout=dropout,
                        **self.kan_params,
                    )
                elif stack_type == "seasonality":
                    net_block = NBEATSSeasonalBlockKAN(
                        units=self.hparams.widths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        min_period=expansion_coefficient_lengths[stack_id],
                        dropout=dropout,
                        **self.kan_params,
                    )
                elif stack_type == "trend":
                    net_block = NBEATSTrendBlockKAN(
                        units=self.hparams.widths[stack_id],
                        thetas_dim=self.hparams.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        dropout=dropout,
                        **self.kan_params,
                    )
                else:
                    raise ValueError(f"Unknown stack type {stack_type}")

                self.net_blocks.append(net_block)

    def update_kan_grid(self):
        """
        Updates grid of KAN layers when using KAN layers in NBEATSBlock.
        WARNING: This relies on 'self.outputs' stored during the last forward pass.
        Ensure this is called immediately after a TRAINING forward pass.
        """
        if not self.training:
            return

        for block in self.net_blocks:
            # updation logic taken from
            # https://github.com/KindXiaoming/pykan/blob/master/kan/MultKAN.py#L2682
            for i, layer in enumerate(block.fc):
                # update basis KAN layers' grid
                layer.update_grid_from_samples(block.outputs[i])
            # update theta backward and theta forward KAN layers' grid
            block.theta_b_fc.update_grid_from_samples(block.outputs[i + 1])
            block.theta_f_fc.update_grid_from_samples(block.outputs[i + 1])
