"""
Implementation of ``nn.Modules`` for N-Beats model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_forecasting.layers._kan._kan_layer import KANLayer
from pytorch_forecasting.layers._nbeats._utils import linear, linspace


class NBEATSBlock(nn.Module):
    """
    Initialize an N-BEATS block using either MLP or KAN layers.

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
    kan_params : dict
        Dictionary of parameters for KAN layers. Only required if `use_kan=True`.
        Default values will be used if not provided. Includes:
            - num : int, default=5
                Number of grid intervals.
            - k : int, default=3
                Order of piecewise polynomial.
            - noise_scale : float, default=0.5
                Initialization noise scale.
            - scale_base_mu : float, default=0.0
                Mean for residual function initialization.
            - scale_base_sigma : float, default=1.0
                Std deviation for residual function initialization.
            - scale_sp : float, default=1.0
                Scale for the spline function.
            - base_fun : nn.Module, default=torch.nn.SiLU()
                Base function module.
            - grid_eps : float, default=0.02
                Determines grid spacing (0 for quantile, 1 for uniform).
            - grid_range : list of float, default=[-1, 1]
                Range of the spline grid.
            - sp_trainable : bool, default=True
                Whether scale_sp is trainable.
            - sb_trainable : bool, default=True
                Whether scale_base is trainable.
            - sparse_init : bool, default=False
                Whether to apply sparse initialization.
    use_kan : bool
        If True, uses KAN layers instead of MLP. Default is False.
    """

    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        dropout=0.1,
        kan_params=None,
        use_kan=False,
    ):
        super().__init__()

        if use_kan and kan_params is None:
            # Define default parameters for KAN if not provided
            kan_params = dict(
                num=5,
                k=3,
                noise_scale=0.5,
                scale_base_mu=0.0,
                scale_base_sigma=1.0,
                scale_sp=1.0,
                base_fun=torch.nn.SiLU(),
                grid_eps=0.02,
                grid_range=[-1, 1],
                sp_trainable=True,
                sb_trainable=True,
                sparse_init=False,
            )

        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.kan_params = kan_params
        self.use_kan = use_kan

        if self.use_kan:
            layers = [
                KANLayer(
                    in_dim=backcast_length,
                    out_dim=units,
                    **self.kan_params,
                )
            ]

            # Add additional layers for deeper structure
            for _ in range(num_block_layers - 1):
                layers.append(
                    KANLayer(
                        in_dim=units,
                        out_dim=units,
                        **self.kan_params,
                    )
                )

            # Define the fully connected layers
            self.fc = nn.Sequential(*layers)

            # Define the theta layers
            self.theta_f_fc = self.theta_b_fc = KANLayer(
                in_dim=units,
                out_dim=thetas_dim,
                **self.kan_params,
            )

        else:
            fc_stack = [
                nn.Linear(backcast_length, units),
                nn.ReLU(),
            ]
            for _ in range(num_block_layers - 1):
                fc_stack.extend([linear(units, units, dropout=dropout), nn.ReLU()])
            self.fc = nn.Sequential(*fc_stack)
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        """
        Forward pass through the block using either MLP or KAN layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after processing through the block.
        """
        if self.use_kan:
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
        return self.fc(x)


class NBEATSSeasonalBlock(NBEATSBlock):
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
    kan_params : dict
        Dictionary of KAN layer parameters. See NBEATSBlock for details.
    use_kan : bool
        If True, uses KAN instead of MLP. Default is False.
    """

    def __init__(
        self,
        units,
        thetas_dim=None,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        nb_harmonics=None,
        min_period=1,
        dropout=0.1,
        kan_params=None,
        use_kan=False,
    ):
        if nb_harmonics:
            thetas_dim = nb_harmonics
        else:
            thetas_dim = forecast_length
        self.min_period = min_period

        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            dropout=dropout,
            kan_params=kan_params,
            use_kan=use_kan,
        )

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

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
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
        amplitudes_backward = self.theta_b_fc(x)
        backcast = amplitudes_backward.mm(self.S_backcast)
        amplitudes_forward = self.theta_f_fc(x)
        forecast = amplitudes_forward.mm(self.S_forecast)

        return backcast, forecast

    def get_frequencies(self, n):
        """
        Generates frequency values based on the backcast and forecast lengths.
        """
        return np.linspace(
            0, (self.backcast_length + self.forecast_length) / self.min_period, n
        )


class NBEATSTrendBlock(NBEATSBlock):
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
    kan_params : dict
        KAN layer parameters. See NBEATSBlock for details.
    use_kan : bool
        If True, uses KAN instead of MLP. Default is False.
    """

    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        dropout=0.1,
        kan_params=None,
        use_kan=False,
    ):
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            dropout=dropout,
            kan_params=kan_params,
            use_kan=use_kan,
        )

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

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
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
        backcast = self.theta_b_fc(x).mm(self.T_backcast)
        forecast = self.theta_f_fc(x).mm(self.T_forecast)
        return backcast, forecast


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
    kan_params : dict
        KAN layer parameters. See NBEATSBlock for details.
    use_kan : bool
        If True, uses KAN instead of MLP. Default is False.
    """

    def __init__(
        self,
        units,
        thetas_dim,
        num_block_layers=4,
        backcast_length=10,
        forecast_length=5,
        dropout=0.1,
        kan_params=None,
        use_kan=False,
    ):
        super().__init__(
            units=units,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            backcast_length=backcast_length,
            forecast_length=forecast_length,
            dropout=dropout,
            kan_params=kan_params,
            use_kan=use_kan,
        )

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
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
