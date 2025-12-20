"""
Implementation of ``nn.Modules`` for N-Beats model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_forecasting.layers._kan._kan_layer import KANLayer
from pytorch_forecasting.layers._nbeats._utils import linear, linspace


class SeasonalMixin:
    """
    Mixin for Seasonal N-BEATS blocks.
    This mixin provides the mechanism to initialize and compute the seasonal component
    using Fourier basis functions.

    Attributes
    ----------
    backcast_length : int
        Length of the input (past) sequence.
    forecast_length : int
        Length of the output (future) sequence.
    min_period : int
        Minimum period for seasonality.
    S_backcast : torch.Tensor
        Backcast side of the seasonality basis matrix.
    S_forecast : torch.Tensor
        Forecast side of the seasonality basis matrix.
    """

    def _init_seasonal(
        self, backcast_length, forecast_length, thetas_dim, min_period
    ):
        """
        Initialize seasonal backcast and forecast coefficients.
        """
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.min_period = min_period

        backcast_linspace, forecast_linspace = linspace(
            backcast_length, forecast_length, centered=False
        )

        p1, p2 = (
            (thetas_dim // 2, thetas_dim // 2)
            if thetas_dim % 2 == 0
            else (thetas_dim // 2, thetas_dim // 2 + 1)
        )
        s1_b = torch.tensor(
            np.cos(2 * np.pi * self.get_frequencies(p1)[:, None] * backcast_linspace),
            dtype=torch.float32,
        )  # H/2-1
        s2_b = torch.tensor(
            np.sin(2 * np.pi * self.get_frequencies(p2)[:, None] * backcast_linspace),
            dtype=torch.float32,
        )
        self.register_buffer("S_backcast", torch.cat([s1_b, s2_b]))

        s1_f = torch.tensor(
            np.cos(2 * np.pi * self.get_frequencies(p1)[:, None] * forecast_linspace),
            dtype=torch.float32,
        )  # H/2-1
        s2_f = torch.tensor(
            np.sin(2 * np.pi * self.get_frequencies(p2)[:, None] * forecast_linspace),
            dtype=torch.float32,
        )
        self.register_buffer("S_forecast", torch.cat([s1_f, s2_f]))

    def seasonal_forward(self, x, theta_b_layer, theta_f_layer):
        """
        Compute seasonal backcast and forecast.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        theta_b_layer : nn.Module
            Layer to compute backcast theta coefficients.
        theta_f_layer : nn.Module
            Layer to compute forecast theta coefficients.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Backcast and forecast tensors.
        """
        amplitudes_backward = theta_b_layer(x)
        backcast = amplitudes_backward.mm(self.S_backcast)
        amplitudes_forward = theta_f_layer(x)
        forecast = amplitudes_forward.mm(self.S_forecast)
        return backcast, forecast

    def get_frequencies(self, n: int) -> np.ndarray:
        """
        Generates frequency values based on the backcast and forecast lengths.
        """
        return np.linspace(
            0, (self.backcast_length + self.forecast_length) / self.min_period, n
        )


class TrendMixin:
    """
    Mixin for Trend N-BEATS blocks.
    This mixin provides the mechanism to initialize and compute the trend component
    using polynomial basis functions.

    Attributes
    ----------
    T_backcast : torch.Tensor
        Backcast side of the trend polynomial basis matrix.
    T_forecast : torch.Tensor
        Forecast side of the trend polynomial basis matrix.
    """

    def _init_trend(self, backcast_length, forecast_length, thetas_dim):
        """
        Initialize trend polynomial coefficients.
        """
        backcast_linspace, forecast_linspace = linspace(
            backcast_length, forecast_length, centered=True
        )
        norm = np.sqrt(
            forecast_length / thetas_dim
        )  # ensure range of predictions is comparable to input
        thetas_dims_range = np.array(range(thetas_dim))
        coefficients = torch.tensor(
            backcast_linspace ** thetas_dims_range[:, None],
            dtype=torch.float32,
        )
        self.register_buffer("T_backcast", coefficients * norm)
        coefficients = torch.tensor(
            forecast_linspace ** thetas_dims_range[:, None],
            dtype=torch.float32,
        )
        self.register_buffer("T_forecast", coefficients * norm)

    def trend_forward(self, x, theta_b_layer, theta_f_layer):
        """
        Compute trend backcast and forecast.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        theta_b_layer : nn.Module
            Layer to compute backcast theta coefficients.
        theta_f_layer : nn.Module
            Layer to compute forecast theta coefficients.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Backcast and forecast tensors.
        """
        backcast = theta_b_layer(x).mm(self.T_backcast)
        forecast = theta_f_layer(x).mm(self.T_forecast)
        return backcast, forecast


class NBEATSBlock(nn.Module):
    """
    Initialize an N-BEATS block using MLP layers.

    Parameters
    ----------
    units : int
        Number of units in each layer.
    thetas_dim : int
        Output dimension of the theta layers.
    num_block_layers : int
        Number of hidden layers in the block. Default is 4.
    backcast_length : int
        Length of the input (past) sequence. Default is 10.
    forecast_length : int
        Length of the output (future) sequence. Default is 5.
    dropout : float
        Dropout rate for regularization. Default is 0.1.
    """

    def __init__(
        self,
        units: int,
        thetas_dim: int,
        num_block_layers: int = 4,
        backcast_length: int = 10,
        forecast_length: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        fc_stack = [
            nn.Linear(backcast_length, units),
            nn.ReLU(),
        ]
        for _ in range(num_block_layers - 1):
            fc_stack.extend([linear(units, units, dropout=dropout), nn.ReLU()])
        self.fc = nn.Sequential(*fc_stack)
        self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block using MLP layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after processing through the block.
        """
        return self.fc(x)


class NBEATSBlockKAN(nn.Module):
    """
    Initialize an N-BEATS block using KAN layers.

    Parameters
    ----------
    units : int
        Number of units in each layer.
    thetas_dim : int
        Output dimension of the theta layers.
    num_block_layers : int
        Number of hidden layers in the block. Default is 4.
    backcast_length : int
        Length of the input (past) sequence. Default is 10.
    forecast_length : int
        Length of the output (future) sequence. Default is 5.
    num : int
        Number of grid intervals. Default: 5.
    k : int
        Order of piecewise polynomial. Default: 3.
    noise_scale : float
        Initialization noise scale. Default: 0.5.
    scale_base_mu : float
        Mean for residual function initialization. Default: 0.0.
    scale_base_sigma : float
        Std deviation for residual function initialization. Default: 1.0.
    scale_sp : float
        Scale for the spline function. Default: 1.0.
    base_fun : nn.Module
        Base function module. Default: torch.nn.SiLU().
    grid_eps : float
        Determines grid spacing (0 for quantile, 1 for uniform). Default: 0.02.
    grid_range : list of float
        Range of the spline grid. Default: [-1, 1].
    sp_trainable : bool
        Whether scale_sp is trainable. Default: True.
    sb_trainable : bool
        Whether scale_base is trainable. Default: True.
    sparse_init : bool
        Whether to apply sparse initialization. Default: False.
    """

    def __init__(
        self,
        units: int,
        thetas_dim: int,
        num_block_layers: int = 4,
        backcast_length: int = 10,
        forecast_length: int = 5,
        num: int = 5,
        k: int = 3,
        noise_scale: float = 0.5,
        scale_base_mu: float = 0.0,
        scale_base_sigma: float = 1.0,
        scale_sp: float = 1.0,
        base_fun: nn.Module = None,
        grid_eps: float = 0.02,
        grid_range: list[float] = None,
        sp_trainable: bool = True,
        sb_trainable: bool = True,
        sparse_init: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        if base_fun is None:
            base_fun = torch.nn.SiLU()
        if grid_range is None:
            grid_range = [-1, 1]

        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.dropout = dropout

        # store KAN params for reuse
        self.kan_params = dict(
            num=num,
            k=k,
            noise_scale=noise_scale,
            scale_base_mu=scale_base_mu,
            scale_base_sigma=scale_base_sigma,
            scale_sp=scale_sp,
            base_fun=base_fun,
            grid_eps=grid_eps,
            grid_range=grid_range,
            sp_trainable=sp_trainable,
            sb_trainable=sb_trainable,
            sparse_init=sparse_init,
        )

        layers = [KANLayer(in_dim=backcast_length, out_dim=units, **self.kan_params)]

        # additional layers
        for _ in range(num_block_layers - 1):
            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))
            layers.append(
                KANLayer(in_dim=units, out_dim=units, **self.kan_params)
            )
        self.fc = nn.Sequential(*layers)

        # theta layers used by subclasses
        self.theta_f_fc = self.theta_b_fc = KANLayer(
            in_dim=units,
            out_dim=thetas_dim,
            **self.kan_params,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block using KAN layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after processing through the block.
        """
        # save outputs to be used in updating grid in kan layers during training
        # outputs logic taken from
        # https://github.com/KindXiaoming/pykan/blob/master/kan/MultKAN.py#L2682
        self.outputs = []
        self.outputs.append(x.clone().detach())
        for layer in self.fc:
            x = layer(x)  # Pass data through the current layer
            # storing outputs for updating grids of self.fc when using KAN
            self.outputs.append(x.clone().detach())
        # storing for updating grids of theta_b_fc and theta_f_fc when using KAN
        self.outputs.append(x.clone().detach())
        return x  # Return final output


class NBEATSSeasonalBlock(NBEATSBlock, SeasonalMixin):
    """
    Initialize a Seasonal N-BEATS block with Fourier-based seasonality modeling.

    Parameters
    ----------
    units : int
        Number of units in each hidden layer.
    thetas_dim : int
        Output dimension of theta layers. Inferred from harmonics if not provided.
    num_block_layers : int
        Number of layers in the block. Default is 4.
    backcast_length : int
        Length of the input (past) sequence. Default is 10.
    forecast_length : int
        Length of the output (future) sequence. Default is 5.
    nb_harmonics : int
        Number of harmonics for Fourier features. Default is None.
    min_period : int
        Minimum period for seasonality. Default is 1.
    dropout : float
        Dropout rate. Default is 0.1.
    """

    def __init__(
        self,
        units: int,
        thetas_dim: int = None,
        num_block_layers: int = 4,
        backcast_length: int = 10,
        forecast_length: int = 5,
        nb_harmonics: int = None,
        min_period: int = 1,
        dropout: float = 0.1,
    ):
        if nb_harmonics:
            thetas_dim = nb_harmonics
        else:
            thetas_dim = forecast_length

        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            dropout=dropout,
        )
        self._init_seasonal(backcast_length, forecast_length, thetas_dim, min_period)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute seasonal backcast and forecast outputs using input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, backcast_length).

        Returns
        -------
        tuple of torch.Tensor
            Tuple (backcast, forecast), each of shape (batch_size, time_steps).
        """
        x = super().forward(x)
        return self.seasonal_forward(x, self.theta_b_fc, self.theta_f_fc)


class NBEATSSeasonalBlockKAN(NBEATSBlockKAN, SeasonalMixin):
    """
    Initialize a Seasonal N-BEATS block using KAN layers.

    Parameters
    ----------
    units : int
        Number of units in each hidden layer.
    thetas_dim : int
        Output dimension of theta layers. Inferred from harmonics if not provided.
    num_block_layers : int
        Number of layers in the block. Default is 4.
    backcast_length : int
        Length of the input (past) sequence. Default is 10.
    forecast_length : int
        Length of the output (future) sequence. Default is 5.
    nb_harmonics : int
        Number of harmonics for Fourier features. Default is None.
    min_period : int
        Minimum period for seasonality. Default is 1.
    num : int
        Number of grid intervals. Default: 5.
    k : int
        Order of piecewise polynomial. Default: 3.
    noise_scale : float
        Initialization noise scale.
    scale_base_mu : float
        Mean for residual function initialization.
    scale_base_sigma : float
        Std deviation for residual function initialization.
    scale_sp : float
        Scale for the spline function.
    base_fun : nn.Module
        Base function module.
    grid_eps : float
        Determines grid spacing.
    grid_range : list of float
        Range of the spline grid.
    sp_trainable : bool
        Whether scale_sp is trainable.
    sb_trainable : bool
        Whether scale_base is trainable.
    sparse_init : bool
        Whether to apply sparse initialization.
    """

    def __init__(
        self,
        units: int,
        thetas_dim: int = None,
        num_block_layers: int = 4,
        backcast_length: int = 10,
        forecast_length: int = 5,
        nb_harmonics: int = None,
        min_period: int = 1,
        dropout: float = 0.1,
        **kan_kwargs,
    ):
        if nb_harmonics:
            thetas_dim = nb_harmonics
        else:
            thetas_dim = forecast_length

        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            dropout=dropout,
            **kan_kwargs,
        )
        self._init_seasonal(backcast_length, forecast_length, thetas_dim, min_period)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute seasonal backcast and forecast outputs using input tensor.
        """
        x = super().forward(x)
        return self.seasonal_forward(x, self.theta_b_fc, self.theta_f_fc)


class NBEATSTrendBlock(NBEATSBlock, TrendMixin):
    """
    Initialize a Trend N-BEATS block using polynomial basis functions.

    Parameters
    ----------
    units : int
        Number of units in each hidden layer.
    thetas_dim : int
        Output dimension of theta layers (number of polynomial terms).
    num_block_layers : int
        Number of hidden layers. Default is 4.
    backcast_length : int
        Length of input sequence. Default is 10.
    forecast_length : int
        Length of output sequence. Default is 5.
    dropout : float
        Dropout rate. Default is 0.1.
    """

    def __init__(
        self,
        units: int,
        thetas_dim: int,
        num_block_layers: int = 4,
        backcast_length: int = 10,
        forecast_length: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            dropout=dropout,
        )
        self._init_trend(backcast_length, forecast_length, thetas_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute backcast and forecast outputs using input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, backcast_length).

        Returns
        -------
        tuple of torch.Tensor
            Tuple (backcast, forecast).
        """

        x = super().forward(x)
        return self.trend_forward(x, self.theta_b_fc, self.theta_f_fc)


class NBEATSTrendBlockKAN(NBEATSBlockKAN, TrendMixin):
    """
    Initialize a Trend N-BEATS block using KAN layers.

    Parameters
    ----------
    units : int
        Number of units in each hidden layer.
    thetas_dim : int
        Output dimension of theta layers (number of polynomial terms).
    num_block_layers : int
        Number of hidden layers. Default is 4.
    backcast_length : int
        Length of input sequence. Default is 10.
    forecast_length : int
        Length of output sequence. Default is 5.
    num : int
        Number of grid intervals. Default: 5.
    k : int
        Order of piecewise polynomial. Default: 3.
    noise_scale : float
        Initialization noise scale.
    scale_base_mu : float
        Mean for residual function initialization.
    scale_base_sigma : float
        Std deviation for residual function initialization.
    scale_sp : float
        Scale for the spline function.
    base_fun : nn.Module
        Base function module.
    grid_eps : float
        Determines grid spacing.
    grid_range : list of float
        Range of the spline grid.
    sp_trainable : bool
        Whether scale_sp is trainable.
    sb_trainable : bool
        Whether scale_base is trainable.
    sparse_init : bool
        Whether to apply sparse initialization.
    """

    def __init__(
        self,
        units: int,
        thetas_dim: int,
        num_block_layers: int = 4,
        backcast_length: int = 10,
        forecast_length: int = 5,
        dropout: float = 0.1,
        **kan_kwargs,
    ):
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            dropout=dropout,
            **kan_kwargs,
        )
        self._init_trend(backcast_length, forecast_length, thetas_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute backcast and forecast outputs using input tensor.
        """
        x = super().forward(x)
        return self.trend_forward(x, self.theta_b_fc, self.theta_f_fc)


class NBEATSGenericBlock(NBEATSBlock):
    """
    Initialize a Generic N-BEATS block using linear mapping of theta outputs.

    Parameters
    ----------
    units : int
        Number of units in each hidden layer.
    thetas_dim : int
        Dimension of the theta parameter.
    num_block_layers : int
        Number of hidden layers. Default is 4.
    backcast_length : int
        Length of past input. Default is 10.
    forecast_length : int
        Length of future prediction. Default is 5.
    dropout : float
        Dropout rate. Default is 0.1.
    """

    def __init__(
        self,
        units: int,
        thetas_dim: int,
        num_block_layers: int = 4,
        backcast_length: int = 10,
        forecast_length: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            dropout=dropout,
        )

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute backcast and forecast using using input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, backcast_length).

        Returns
        -------
        tuple of torch.Tensor
            Tuple (backcast, forecast).
        """
        x = super().forward(x)
        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))
        return self.backcast_fc(theta_b), self.forecast_fc(theta_f)


class NBEATSGenericBlockKAN(NBEATSBlockKAN):
    """
    Initialize a Generic N-BEATS block using KAN layers.

    Parameters
    ----------
    units : int
        Number of units in each hidden layer.
    thetas_dim : int
        Dimension of the theta parameter.
    num_block_layers : int
        Number of hidden layers. Default is 4.
    backcast_length : int
        Length of past input. Default is 10.
    forecast_length : int
        Length of future prediction. Default is 5.
    num : int
        Number of grid intervals. Default: 5.
    k : int
        Order of piecewise polynomial. Default: 3.
    noise_scale : float
        Initialization noise scale.
    scale_base_mu : float
        Mean for residual function initialization.
    scale_base_sigma : float
        Std deviation for residual function initialization.
    scale_sp : float
        Scale for the spline function.
    base_fun : nn.Module
        Base function module.
    grid_eps : float
        Determines grid spacing.
    grid_range : list of float
        Range of the spline grid.
    sp_trainable : bool
        Whether scale_sp is trainable.
    sb_trainable : bool
        Whether scale_base is trainable.
    sparse_init : bool
        Whether to apply sparse initialization.
    """

    def __init__(
        self,
        units: int,
        thetas_dim: int,
        num_block_layers: int = 4,
        backcast_length: int = 10,
        forecast_length: int = 5,
        dropout: float = 0.1,
        **kan_kwargs,
    ):
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            dropout=dropout,
            **kan_kwargs,
        )

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute backcast and forecast using using input tensor.
        """
        x = super().forward(x)
        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))
        return self.backcast_fc(theta_b), self.forecast_fc(theta_f)
